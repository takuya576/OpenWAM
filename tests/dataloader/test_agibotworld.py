"""Clean AgiBotWorld exports retain physical semantics without read-time filtering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.agibotworld import (
    _GRIPPER_CONTRACT,
    AgiBotWorldDataset,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError

FPS = 15.0
HEAD = "observation.images.head"
LEFT = "observation.images.hand_left"
RIGHT = "observation.images.hand_right"
CAMS = (HEAD, LEFT, RIGHT)
NUM_FRAMES = 33
VIDEO_STRIDE = 4


def _make_bucket(
    bucket: Path,
    episodes: list[dict],
    *,
    action_velocity=None,
    state_velocity=None,
    action_gripper=None,
    state_gripper=None,
    action_dex=None,
    state_dex=None,
) -> Path:
    """Write a minimal grippered AgiBotWorld bucket.

    ``episodes`` items contain each exported episode length. All
    episodes live in one data shard and one video file per camera, so the
    per-episode offsets are plain cumulative sums — which is exactly the layout
    that makes an offset bug read into the NEXT episode instead of erroring.
    """
    meta = bucket / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": FPS,
                "robot_type": "g2a",
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {},
            }
        )
    )
    rows, cum = [], 0
    for ep, spec in enumerate(episodes):
        row = {
            "episode_index": ep,
            "length": int(spec["length"]),
            "tasks": ["do the thing"],
            "dataset_from_index": cum,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for cam in CAMS:
            row[f"videos/{cam}/chunk_index"] = 0
            row[f"videos/{cam}/file_index"] = 0
        rows.append(row)
        cum += int(spec["length"])
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), meta / "episodes" / "chunk-000.parquet")

    pd.DataFrame({"task_index": [0]}, index=pd.Index(["do the thing"], name="task")).to_parquet(meta / "tasks.parquet")

    # data shard: ee_base[0] encodes the GLOBAL row index so a misaligned slice
    # is detectable from the action payload alone.
    total = cum

    def _values(value, dim):
        if value is None:
            return np.zeros((total, dim), dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (dim,):
            arr = np.broadcast_to(arr, (total, dim)).copy()
        if arr.shape != (total, dim):
            raise ValueError(f"expected {(total, dim)}, got {arr.shape}")
        return arr

    action_vel = _values(action_velocity, 3)
    state_vel = _values(state_velocity, 3)
    action_grip = _values(action_gripper, 2)
    state_grip = _values(state_gripper, 2)
    action_fingers = _values(action_dex, 12)
    state_fingers = _values(state_dex, 12)
    (meta / "stats.json").write_text(
        json.dumps(
            {
                "action.robot_velocity": {
                    "min": action_vel.min(axis=0).tolist(),
                    "max": action_vel.max(axis=0).tolist(),
                },
                "observation.state.robot_velocity": {
                    "min": state_vel.min(axis=0).tolist(),
                    "max": state_vel.max(axis=0).tolist(),
                },
            }
        )
    )

    gidx = np.arange(total, dtype=np.float32)
    ee = np.zeros((total, 18), dtype=np.float32)
    ee[:, 0] = gidx
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_index = np.concatenate([np.arange(int(s["length"]), dtype=np.int64) for s in episodes])
    ep_index = np.concatenate([np.full(int(s["length"]), e, dtype=np.int64) for e, s in enumerate(episodes)])
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "task_index": np.zeros(total, dtype=np.int64),
                    "episode_index": ep_index,
                    "frame_index": frame_index,
                    "action.ee_base": list(ee),
                    "observation.state.ee_base": list(ee),
                    "action.gripper": list(action_grip),
                    "observation.state.gripper": list(state_grip),
                    "action.dex": list(action_fingers),
                    "observation.state.dex": list(state_fingers),
                    "action.robot_velocity": list(action_vel),
                    "observation.state.robot_velocity": list(state_vel),
                }
            )
        ),
        data_dir / "file-000.parquet",
    )
    return bucket


@pytest.fixture
def patch_decode(monkeypatch):
    """Encode each requested frame index into pixel 0 so the decoded frames can be
    mapped back to the file-local indices the reader asked for."""

    def fake_decode(path, frame_indices, height, width):
        out = []
        for fi in frame_indices:
            im = Image.new("RGB", (width, height), (0, 0, 0))
            im.putpixel((0, 0), (int(fi) % 256, (int(fi) // 256) % 256, 7))
            out.append(im)
        return out

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)


def _decoded_index(canvas: Image.Image) -> int:
    r, g, _ = canvas.getpixel((0, 0))
    return r + 256 * g


def _reader(bucket: Path, **kw) -> AgiBotWorldDataset:
    return AgiBotWorldDataset(
        dataset_dir=str(bucket),
        num_frames=NUM_FRAMES,
        video_stride=VIDEO_STRIDE,
        height=384,
        width=320,
        multiview=True,
        unify_action=False,
        **kw,
    )


class TestPhysicalSemantics:
    @staticmethod
    def _unified_reader(bucket: Path) -> AgiBotWorldDataset:
        return AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=NUM_FRAMES,
            video_stride=VIDEO_STRIDE,
            height=384,
            width=320,
            multiview=True,
            unify_action=True,
            normalize_mode=None,
        )

    def test_command_only_base_motion_keeps_action_and_masks_proprio(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 40, "flag": 0, "delta": 0}],
            action_velocity=[1.25, 0.0, -0.5],
            state_velocity=[0.0, 0.0, 0.0],
            action_gripper=[0.0, 1.0],  # source: left open, right closed
            state_gripper=[0.035, 0.125],  # source actuator: left open, right closed
        )
        sample = self._unified_reader(bucket)[0]

        # Source action 0=open / 1=closed is inverted into 0=closed / 1=open.
        assert sample["action"][0, 9].item() == pytest.approx(1.0)
        assert sample["action"][0, 43].item() == pytest.approx(0.0)
        # State closing-actuator endpoints become the same canonical aperture fraction.
        assert sample["proprio"][0, 9].item() == pytest.approx(1.0)
        assert sample["proprio"][0, 43].item() == pytest.approx(0.0)

        assert sample["action_mask"][:, 68].all()
        assert sample["action_mask"][:, 70].all()
        assert not sample["proprio_mask"][:, 68].any()
        assert not sample["proprio_mask"][:, 70].any()
        # Masked values are kept numerically neutral as well.
        assert sample["proprio"][0, 68].item() == 0.0
        assert sample["proprio"][0, 70].item() == 0.0
        assert sample["action"][0, 68].item() == pytest.approx(1.25)
        assert sample["action"][0, 70].item() == pytest.approx(-0.5)

    def test_state_only_base_motion_masks_action_independently(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 40, "flag": 0, "delta": 0}],
            action_velocity=[0.0, 0.0, 0.0],
            state_velocity=[0.25, 0.0, 0.4],
        )
        sample = self._unified_reader(bucket)[0]

        assert not sample["action_mask"][:, 68].any()
        assert not sample["action_mask"][:, 70].any()
        assert sample["proprio_mask"][:, 68].all()
        assert sample["proprio_mask"][:, 70].all()
        assert sample["action"][0, 68].item() == 0.0
        assert sample["action"][0, 70].item() == 0.0
        assert sample["proprio"][0, 68].item() == pytest.approx(0.25)
        assert sample["proprio"][0, 70].item() == pytest.approx(0.4)

    def test_dex_fingers_keep_distinct_left_and_right_unified_slots(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "475",
            [{"length": 40}],
            action_dex=np.arange(12, dtype=np.float32),
            state_dex=np.arange(12, dtype=np.float32) / 10,
        )
        sample = self._unified_reader(bucket)[0]
        np.testing.assert_allclose(sample["action"][0, 10:16], np.arange(6))
        np.testing.assert_allclose(sample["action"][0, 44:50], np.arange(6, 12))
        np.testing.assert_allclose(sample["proprio"][0, 44:50], np.arange(6, 12) / 10)
        assert not sample["action_mask"][:, 9].any()
        assert not sample["action_mask"][:, 43].any()

    def test_unified_path_requires_bucket_motion_stats(self, tmp_path):
        bucket = _make_bucket(tmp_path / "b", [{"length": 40}])
        (bucket / "meta" / "stats.json").unlink()
        with pytest.raises(DataContractError, match="movement statistics"):
            self._unified_reader(bucket)

    def test_non_unified_path_does_not_require_base_motion_stats(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 40, "flag": 0, "delta": 0}],
        )
        (bucket / "meta" / "stats.json").unlink()

        ds = AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=NUM_FRAMES,
            video_stride=VIDEO_STRIDE,
            normalize_mode=None,
            unify_action=False,
        )
        sample = ds[0]
        assert sample["action"].shape[-1] == 20
        assert sample["proprio"].shape[-1] == 20


class TestTemporalAlignment:
    def test_truncated_tail_drops_clamped_final_target(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [{"length": 12, "flag": 0, "delta": 0}],
        )
        ds = AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=9,
            video_stride=4,
            normalize_mode=None,
        )

        assert ds._n_supervised_action_steps(9) == 9
        assert ds._n_supervised_action_steps(4) == 3
        assert ds._n_supervised_action_steps(1) == 0
        # Offset 3 is a full 9-row window: T_action=8 already omits the final
        # source row. Offset 10 has two rows and must supervise only the first.
        assert int(ds[3]["action_mask"].any(dim=1).sum()) == 8
        assert int(ds[len(ds) - 1]["action_mask"].any(dim=1).sum()) == 1

    def test_one_row_episode_is_not_sampleable(self, tmp_path, patch_decode):
        bucket = _make_bucket(
            tmp_path / "b",
            [
                {"length": 1, "flag": 0, "delta": 0},
                {"length": 4, "flag": 0, "delta": 0},
            ],
        )
        ds = AgiBotWorldDataset(
            dataset_dir=str(bucket),
            num_frames=9,
            video_stride=4,
            normalize_mode=None,
        )

        assert ds._train_min_window_len() == 2
        assert len(ds) == 3
        assert all(ds[i]["action_mask"].any() for i in range(len(ds)))


class TestCleanExport:
    def test_every_exported_episode_is_kept_without_annotation_files(self, tmp_path, patch_decode):
        bucket = _make_bucket(tmp_path / "family" / "bucket", [{"length": 20}, {"length": 40}])
        ds = _reader(bucket)
        assert ds._eps_df["episode_index"].tolist() == [0, 1]
        assert ds._eps_df["length"].tolist() == [20, 40]
        assert ds[0]["action"][0, 0].item() == 0
        ep1_start = int(ds._cum_n_starts[1])
        assert ds[ep1_start]["action"][0, 0].item() == 20
        assert _decoded_index(ds[ep1_start]["video"][0]) == 20

    def test_selection_knobs_are_not_part_of_the_config_surface(self):
        assert "use_segment_annotations" not in AgiBotWorldDataset.CONFIG_KEYS
        assert "segment_max_trim_ratio" not in AgiBotWorldDataset.CONFIG_KEYS

    @staticmethod
    def _write_stats(bucket, **overrides):
        def block(dim):
            return {
                "mean": [0.0] * dim,
                "std": [1.0] * dim,
                "min": [-2.0] * dim,
                "max": [2.0] * dim,
                "q01": [-2.0] * dim,
                "q99": [2.0] * dim,
            }

        stats = {"gripper_contract": _GRIPPER_CONTRACT}
        for prefix in ("action", "observation.state"):
            stats[f"{prefix}.ee_base"] = block(18)
            stats[f"{prefix}.gripper"] = block(2)
        stats.update(overrides)
        path = bucket.parent / "meta" / "stats_g2a.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(stats))
        return path

    def test_loads_precomputed_family_root_stats(self, tmp_path, patch_decode):
        bucket = _make_bucket(tmp_path / "family" / "bucket", [{"length": 40}])
        self._write_stats(bucket)
        ds = _reader(bucket, normalize_mode="quantile")
        sample = ds[1]
        assert sample["action"][0, 0].item() == pytest.approx(0.5)
        assert ds._action_norm_stats["q01"][3] == -1
        assert ds._action_norm_stats["q99"][3] == 1

    def test_missing_stats_fail_without_generating_them(self, tmp_path):
        bucket = _make_bucket(tmp_path / "family" / "bucket", [{"length": 40}])
        with pytest.raises(FileNotFoundError, match="stats_g2a.json"):
            _reader(bucket, normalize_mode="quantile")
        assert not (bucket.parent / "meta" / "stats_g2a.json").exists()

    def test_invalid_gripper_stats_contract_is_rejected(self, tmp_path):
        bucket = _make_bucket(tmp_path / "family" / "bucket", [{"length": 40}])
        self._write_stats(bucket, gripper_contract={})
        with pytest.raises(DataContractError, match="gripper_contract"):
            _reader(bucket, normalize_mode="quantile")

    def test_stats_width_and_finiteness_are_validated(self, tmp_path):
        bucket = _make_bucket(tmp_path / "family" / "bucket", [{"length": 40}])
        path = self._write_stats(bucket)
        stats = json.loads(path.read_text())
        stats["action.ee_base"]["mean"] = [0.0]
        path.write_text(json.dumps(stats))
        with pytest.raises((DataContractError, ValueError), match="18|shape"):
            _reader(bucket, normalize_mode="quantile")


def test_each_camera_uses_its_own_timestamp_after_a_video_gap(tmp_path, monkeypatch):
    bucket = _make_bucket(tmp_path / "bucket", [{"length": 8}, {"length": 8}])
    manifest = bucket / "meta" / "episodes" / "chunk-000.parquet"
    episodes = pd.read_parquet(manifest)
    for number, camera in enumerate(CAMS):
        starts = np.array([100, 150]) + number * 100
        episodes[f"videos/{camera}/from_timestamp"] = starts / FPS
        episodes[f"videos/{camera}/to_timestamp"] = (starts + 8) / FPS
    episodes.to_parquet(manifest, index=False)
    requested = {}

    def decode(path, frame_indices, height, width):
        requested[Path(path).parts[-3]] = frame_indices
        return [Image.new("RGB", (width, height)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)
    dataset = AgiBotWorldDataset(
        dataset_dir=str(bucket), normalize_mode=None, multiview=True, num_frames=5, video_stride=1
    )
    sample = dataset._getitem_impl(int(dataset._cum_n_starts[1]))
    assert sample["action"][0, 0].item() == 8
    for number, camera in enumerate(CAMS):
        assert requested[camera] == list(range(150 + number * 100, 155 + number * 100))
