"""YamDataset tests on a synthetic bimanual YAM LeRobot-v2.1 bucket.

Covers the v2.1 episode/file mapping, the joint -> 80-D slot contract, the
yaml split override (held-out episodes stay out of training and stats), the
stats rules (gripper pin, still-joint span floor, split-keyed file, stale-file
rejection), camera placement on the multiview canvas, the round trip
through the deploy normalizer, and delta actions back to absolute through the
deploy engine. Video decoding is mocked (empty mp4 files on
disk); no model, GPU, or real recording is needed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.bases.lerobot_v3_reader import _read_data_table_cached
from openwam.dataloader.registry import DATASET_REGISTRY, build_dataset
from openwam.dataloader.utils.stats_computation.yam_stats_computation import compute_yam_stats
from openwam.dataloader.utils.unify_action import parse_unify_spec
from openwam.dataloader.yam import (
    ACTION_STATS_KEY,
    HEAD_CAMERA,
    JOINT_NAMES,
    LEFT_WRIST_CAMERA,
    MIN_JOINT_SPAN_RAD,
    RIGHT_WRIST_CAMERA,
    STATE_STATS_KEY,
    YamDataset,
    default_stats_path,
)
from openwam.deploy.engine import JointInferenceEngine
from openwam.deploy.model_loader import _build_normalizer, delta_action_dims_from_cfg

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "dataloader" / "yam.yaml"
UNIFY_MAP = ["68-73", 9, "74-79", 43]
MAPPED_SLOTS = [*range(68, 74), 9, *range(74, 80), 43]
TASK = "Use the right arm to place the cube on the blue pad."
LENGTHS = (40, 50, 45)
CAMERA_COLORS = {HEAD_CAMERA: (200, 0, 0), LEFT_WRIST_CAMERA: (0, 200, 0), RIGHT_WRIST_CAMERA: (0, 0, 200)}


def _episode_state(ep: int, length: int) -> np.ndarray:
    """Still left arm (sub-span jitter, open gripper); right arm sweeps and closes."""
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    state = np.zeros((length, 14), dtype=np.float32)
    state[:, 0:6] = 0.01 + 0.001 * np.sin(7 * t + ep)[:, None]
    state[:, 6] = 1.0
    state[:, 7:13] = 0.2 * ep + np.outer(t, np.arange(1, 7, dtype=np.float32))
    state[:, 13] = 1.0 - t
    return state


def _episode_action(ep: int, length: int) -> np.ndarray:
    """Leader command one step ahead of the follower."""
    state = _episode_state(ep, length)
    return np.concatenate([state[1:], state[-1:]], axis=0)


def make_bucket(root: Path, *, train_split: str = "0:2") -> Path:
    bucket = root / "yam_task"
    (bucket / "meta").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    joint_feature = {"dtype": "float32", "shape": [14], "names": JOINT_NAMES}
    features = {"observation.state": joint_feature, "action": joint_feature}
    for camera in CAMERA_COLORS:
        (bucket / "videos" / "chunk-000" / camera).mkdir(parents=True)
        features[camera] = {"dtype": "video", "shape": [480, 640, 3]}
    info = {
        "codebase_version": "v2.1",
        "robot_type": "yam_bimanual",
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "splits": {"train": train_split},
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))

    episodes = []
    for ep, length in enumerate(LENGTHS):
        df = pd.DataFrame(
            {
                "observation.state": list(_episode_state(ep, length)),
                "action": list(_episode_action(ep, length)),
                "frame_index": np.arange(length),
                "episode_index": np.full(length, ep),
                "task_index": np.zeros(length, dtype=np.int64),
            }
        )
        df.to_parquet(bucket / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        for camera in CAMERA_COLORS:
            (bucket / "videos" / "chunk-000" / camera / f"episode_{ep:06d}.mp4").write_bytes(b"")
        episodes.append({"episode_index": ep, "tasks": [TASK], "length": length})
    with (bucket / "meta" / "episodes.jsonl").open("w") as f:
        for row in episodes:
            f.write(json.dumps(row) + "\n")
    (bucket / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    return bucket


@pytest.fixture()
def bucket(tmp_path):
    return make_bucket(tmp_path)


@pytest.fixture()
def decode_calls(monkeypatch):
    """Mock decoding with one solid color per camera; record (path, frame indices)."""
    calls = []

    def fake_decode(path, frame_indices, height, width):
        calls.append((path, list(frame_indices)))
        color = next(c for camera, c in CAMERA_COLORS.items() if f"/{camera}/" in path)
        return [Image.new("RGB", (width, height), color) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)
    return calls


@pytest.fixture(autouse=True)
def _clear_parquet_cache():
    _read_data_table_cached.cache_clear()
    yield
    _read_data_table_cached.cache_clear()


def _make_ds(bucket, **kw):
    kw.setdefault("multiview", True)
    kw.setdefault("unify_action", True)
    kw.setdefault("unify_action_map", UNIFY_MAP)
    return YamDataset(str(bucket), **kw)


def test_config_registry_and_slot_map():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["type"] == "yam"
    assert config["action_mode"] == "joint"
    assert config["normalize_mode"] == "min-max"
    assert DATASET_REGISTRY["yam"] is YamDataset
    assert parse_unify_spec(config["unify_action_map"], 80).tolist() == MAPPED_SLOTS
    # The held-out open-loop episodes must never reach training.
    train_end = int(config["splits"]["train"].split(":")[1])
    val_start = int(config["splits"]["val"].split(":")[0])
    assert train_end <= val_start


def test_stats_pin_grippers_and_floor_still_joints(bucket):
    payload = compute_yam_stats(bucket)
    train_state = np.concatenate([_episode_state(ep, LENGTHS[ep]) for ep in (0, 1)])
    for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
        block = payload[key]
        assert block["num_timesteps"] == LENGTHS[0] + LENGTHS[1]  # train split only
        mn, mx = np.asarray(block["min"]), np.asarray(block["max"])
        np.testing.assert_allclose(mn[[6, 13]], 0.0)
        np.testing.assert_allclose(mx[[6, 13]], 1.0)
        np.testing.assert_allclose(mx[0:6] - mn[0:6], MIN_JOINT_SPAN_RAD, rtol=1e-6)
    state_block = payload[STATE_STATS_KEY]
    np.testing.assert_allclose(
        0.5 * (np.asarray(state_block["min"][0:6]) + np.asarray(state_block["max"][0:6])),
        0.5 * (train_state[:, 0:6].min(0) + train_state[:, 0:6].max(0)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(state_block["min"][7:13], train_state[:, 7:13].min(0), rtol=1e-6)
    np.testing.assert_allclose(state_block["max"][7:13], train_state[:, 7:13].max(0), rtol=1e-6)


def test_sample_contract(bucket, decode_calls):
    ds = _make_ds(bucket)
    assert len(ds) == LENGTHS[0] + LENGTHS[1]
    assert ds.action_dim == 80
    assert ds.normalization_stats_path == str(bucket / "meta" / "yam_normalization_stats_train_0-2.npy")

    sample = ds[0]
    assert sample["prompt"] == TASK
    assert tuple(sample["action"].shape) == (32, 80)
    assert tuple(sample["proprio"].shape) == (1, 80)
    expected_mask = np.zeros(80, dtype=bool)
    expected_mask[MAPPED_SLOTS] = True
    assert (sample["action_mask"].numpy() == expected_mask).all()
    assert (sample["proprio_mask"].numpy()[0] == expected_mask).all()
    action = sample["action"].numpy()
    assert np.abs(action).max() <= 1.0
    assert not action[:, ~expected_mask].any()
    np.testing.assert_allclose(action[:, 9], 1.0)  # left gripper open -> +1
    assert sample["proprio"][0, 43] == pytest.approx(1.0)  # right gripper starts open
    assert np.abs(action[:, 68:74]).max() < 0.05  # still left arm stays near 0

    assert len(sample["video"]) == 9
    canvas = sample["video"][0]
    assert canvas.size == (320, 384)
    assert canvas.getpixel((160, 100)) == CAMERA_COLORS[HEAD_CAMERA]
    assert canvas.getpixel((80, 320)) == CAMERA_COLORS[LEFT_WRIST_CAMERA]
    assert canvas.getpixel((240, 320)) == CAMERA_COLORS[RIGHT_WRIST_CAMERA]


def test_window_reads_its_episode_files(bucket, decode_calls):
    ds = _make_ds(bucket)
    sample = ds[LENGTHS[0] + 5]  # episode 1, window starting at frame 5
    paths = sorted(Path(path).relative_to(bucket).as_posix() for path, _ in decode_calls)
    assert paths == sorted(f"videos/chunk-000/{camera}/episode_000001.mp4" for camera in CAMERA_COLORS)
    assert all(indices == list(range(5, 38, 4)) for _, indices in decode_calls)
    stats = ds._state_normalization_stats
    raw = _episode_state(1, LENGTHS[1])[5, 7]  # right_joint_1 -> slot 74
    expected = 2 * (raw - stats["min"][7]) / (stats["max"][7] - stats["min"][7]) - 1
    assert sample["proprio"][0, 74] == pytest.approx(expected, abs=1e-5)


def test_deploy_normalizer_inverts_training(bucket, decode_calls, tmp_path):
    ds = _make_ds(bucket)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    shutil.copyfile(ds.normalization_stats_path, ckpt_dir / "normalization_stats.npy")
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "joint",
                "unify_action": True,
                "unify_action_map": UNIFY_MAP,
            }
        }
    )
    normalizer = _build_normalizer(cfg, str(ckpt_dir))

    sample = ds[LENGTHS[0] + 3]
    raw_action = _episode_action(1, LENGTHS[1])[3:35]
    np.testing.assert_allclose(normalizer.unnormalize(sample["action"].numpy()), raw_action, rtol=1e-5, atol=1e-5)
    raw_state = _episode_state(1, LENGTHS[1])[3:4]
    np.testing.assert_allclose(normalizer.normalize(raw_state), sample["proprio"].numpy(), atol=1e-6)


DELTA_DIMS = ["0-5", "7-12"]
ARM_DIMS = [*range(0, 6), *range(7, 13)]


class _Scheduler:
    num_train_timesteps = 1000

    def set_timesteps(self, n, shift=5.0):
        self.timesteps = self.sigmas = torch.linspace(1.0, 0.0, n)


class _ReplayArchitecture:
    """Stands in for the model: emits one training sample's action chunk through the deploy normalizer."""

    def __init__(self, normalizer, action, cfg):
        self.normalizer = normalizer
        self.video_scheduler = self.action_scheduler = _Scheduler()
        self.delta_action_dims = delta_action_dims_from_cfg(cfg)  # as load_from_checkpoint_dir binds it
        self._action = action

    def apply_compile_optimizations(self, compile_cfg):
        del compile_cfg

    def normalize_deploy_proprio(self, proprio):
        return torch.from_numpy(self.normalizer.normalize(np.asarray(proprio, dtype=np.float32)))

    def generate(self, **kwargs):
        del kwargs
        return {"video": None, "actions": self.normalizer.unnormalize(self._action)}


def test_delta_actions_round_trip_to_absolute_through_the_engine(bucket, decode_calls, tmp_path):
    """A delta training target, served by the engine for its window's state, is the recorded absolute action."""
    ds = _make_ds(bucket, delta_action_dims=DELTA_DIMS)
    assert Path(ds.normalization_stats_path).name == "yam_normalization_stats_train_0-2_delta32.npy"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    shutil.copyfile(ds.normalization_stats_path, ckpt_dir / "normalization_stats.npy")
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "normalize_mode": "min-max",
                "action_mode": "joint",
                "unify_action": True,
                "unify_action_map": UNIFY_MAP,
                "delta_action_dims": DELTA_DIMS,
            },
            "inference": {"denoise_steps": 2, "denoise_mode": "sync"},
            "optimization": {"decode_video": False},
        }
    )
    normalizer = _build_normalizer(cfg, str(ckpt_dir))

    sample = ds[LENGTHS[0] + 3]
    raw_state = _episode_state(1, LENGTHS[1])[3]
    raw_action = _episode_action(1, LENGTHS[1])[3:35]
    relative = normalizer.unnormalize(sample["action"].numpy())
    np.testing.assert_allclose(relative[:, ARM_DIMS], raw_action[:, ARM_DIMS] - raw_state[ARM_DIMS], atol=1e-5)
    np.testing.assert_allclose(relative[:, [6, 13]], raw_action[:, [6, 13]], atol=1e-6)  # grippers stay absolute

    engine = JointInferenceEngine(cfg=cfg, architecture=_ReplayArchitecture(normalizer, sample["action"].numpy(), cfg))
    actions = engine.generate({"observation": {"state": raw_state}, "num_frames": 33})["actions"]
    np.testing.assert_allclose(actions, raw_action, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="relative to the robot state"):
        engine.generate({"num_frames": 33})


def test_delta_stats_cover_every_chunk_target(bucket):
    payload = compute_yam_stats(bucket, delta_action_dims=DELTA_DIMS, delta_horizon=32)
    block = payload[ACTION_STATS_KEY]
    assert block["delta_action_dims"] == ARM_DIMS
    assert block["delta_horizon"] == 32
    targets = []
    for ep in (0, 1):
        state, action = _episode_state(ep, LENGTHS[ep]), _episode_action(ep, LENGTHS[ep])
        for t in range(LENGTHS[ep]):  # every window start, as the reader samples them
            chunk = action[t : t + 32].copy()
            chunk[:, ARM_DIMS] -= state[t, ARM_DIMS]
            targets.append(chunk)
    targets = np.concatenate(targets)
    assert block["num_timesteps"] == len(targets)
    np.testing.assert_allclose(block["min"][7:13], targets[:, 7:13].min(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(block["max"][7:13], targets[:, 7:13].max(0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.subtract(block["max"], block["min"])[0:6], MIN_JOINT_SPAN_RAD, rtol=1e-6)
    np.testing.assert_allclose(block["min"][13], 0.0)
    np.testing.assert_allclose(block["max"][13], 1.0)
    # Proprio stays absolute.
    absolute = compute_yam_stats(bucket)
    assert payload[STATE_STATS_KEY]["max"] == absolute[STATE_STATS_KEY]["max"]


def test_stats_from_other_delta_settings_are_rejected(bucket):
    absolute = _make_ds(bucket).normalization_stats_path
    with pytest.raises(ValueError, match="fitted on actions with"):
        _make_ds(bucket, delta_action_dims=DELTA_DIMS, normalization_stats_path=absolute)
    delta = _make_ds(bucket, delta_action_dims=DELTA_DIMS).normalization_stats_path
    assert delta != absolute
    with pytest.raises(ValueError, match="fitted on actions with"):
        _make_ds(bucket, normalization_stats_path=delta)
    with pytest.raises(ValueError, match="fitted on actions with"):
        _make_ds(bucket, delta_action_dims=["7-12"], normalization_stats_path=delta)


def test_delta_dims_must_leave_grippers_absolute(bucket):
    with pytest.raises(ValueError, match="grippers"):
        _make_ds(bucket, delta_action_dims=["0-6"])


def test_build_dataset_from_config(bucket, decode_calls):
    config = OmegaConf.load(CONFIG_PATH)
    config.dataset_dir = str(bucket)
    ds = build_dataset(config, split="train")
    assert isinstance(ds, YamDataset)
    assert ds.action_dim == 80
    assert ds[0]["action"].shape[-1] == 80


def test_explicit_layout_controls_decoded_cameras(bucket, decode_calls):
    ds = _make_ds(bucket, camera_layout=[HEAD_CAMERA, None, RIGHT_WRIST_CAMERA])
    canvas = ds[0]["video"][0]
    assert canvas.getpixel((80, 320)) == (0, 0, 0)
    assert canvas.getpixel((240, 320)) == CAMERA_COLORS[RIGHT_WRIST_CAMERA]
    assert not any(f"/{LEFT_WRIST_CAMERA}/" in path for path, _ in decode_calls)


def test_split_override_holds_out_eval_episodes(bucket, decode_calls):
    splits = {"train": "0:1", "val": "2:3"}  # episode 1 in neither split
    train = _make_ds(bucket, splits=splits)
    assert len(train) == LENGTHS[0]
    assert Path(train.normalization_stats_path).name == "yam_normalization_stats_train_0-1.npy"
    payload = np.load(train.normalization_stats_path, allow_pickle=True).item()
    assert payload[ACTION_STATS_KEY]["num_timesteps"] == LENGTHS[0]
    assert payload[ACTION_STATS_KEY]["train_split"] == "0:1"
    np.testing.assert_allclose(payload[STATE_STATS_KEY]["max"][7:13], _episode_state(0, LENGTHS[0])[:, 7:13].max(0))

    val = _make_ds(bucket, splits=splits, split="val")
    assert len(val) == LENGTHS[2] - 33 + 1  # full windows only
    assert val.normalization_stats_path == train.normalization_stats_path
    val[0]
    assert all("episode_000002.mp4" in path for path, _ in decode_calls[-3:])


def test_episode_list_split_trains_on_the_subset(bucket, decode_calls):
    splits = OmegaConf.create({"train": [2, 0], "val": "1:2"})  # as Hydra passes a yaml list
    train = _make_ds(bucket, splits=splits)
    assert len(train) == LENGTHS[0] + LENGTHS[2]
    assert Path(train.normalization_stats_path).name.startswith("yam_normalization_stats_train_2eps_")
    payload = np.load(train.normalization_stats_path, allow_pickle=True).item()
    assert payload[ACTION_STATS_KEY]["train_split"] == [0, 2]  # sorted, plain ints
    assert payload[ACTION_STATS_KEY]["num_timesteps"] == LENGTHS[0] + LENGTHS[2]

    # Written in another order, the same subset reuses the same stats.
    assert _make_ds(bucket, splits={"train": [0, 2]}).normalization_stats_path == train.normalization_stats_path
    assert default_stats_path(bucket, {"train": [0, 1]}) != default_stats_path(bucket, {"train": [0, 2]})

    val = _make_ds(bucket, splits=splits, split="val")
    assert len(val) == LENGTHS[1] - 33 + 1
    assert val.normalization_stats_path == train.normalization_stats_path


@pytest.mark.parametrize(
    "train, match", [([0, 0], "duplicates"), ([0, 7], r"lists episodes \[7\] that the dataset does not have")]
)
def test_bad_episode_lists_are_rejected(bucket, train, match):
    with pytest.raises(ValueError, match=match):
        _make_ds(bucket, splits={"train": train})


def test_place_cube_right_sweep_subsets_are_nested_per_position():
    """n<N> takes the first N/14 demos of each position (episodes 4p..4p+3), all outside val."""
    subsets = {}
    for n in (14, 28, 42, 56):
        config = OmegaConf.load(CONFIG_PATH.parent / f"yam_place_cube_right_n{n}.yaml")
        assert list(config.defaults) == ["yam", "_self_"]
        train = config.splits.train
        subsets[n] = set(range(*map(int, train.split(":")))) if isinstance(train, str) else set(train)
        assert subsets[n] == {4 * p + i for p in range(14) for i in range(n // 14)}
        assert not subsets[n] & set(range(*map(int, config.splits.val.split(":"))))
    assert subsets[14] < subsets[28] < subsets[42] < subsets[56]


def test_stats_from_another_split_are_rejected(bucket):
    stats_path = _make_ds(bucket).normalization_stats_path  # fitted on info.json's 0:2
    with pytest.raises(ValueError, match="fitted on train split '0:2'"):
        _make_ds(bucket, splits={"train": "0:1"}, normalization_stats_path=stats_path)


def test_stats_from_re_recorded_data_are_rejected(bucket):
    _make_ds(bucket)
    # Re-record episode 1 with the same length but different joint values.
    path = bucket / "data" / "chunk-000" / "episode_000001.parquet"
    df = pd.read_parquet(path)
    df["action"] = [row * 1.5 for row in df["action"]]
    df.to_parquet(path)
    with pytest.raises(ValueError, match="different train-split data"):
        _make_ds(bucket)


def test_held_out_episodes_do_not_affect_stats_fingerprint(bucket):
    _make_ds(bucket)  # train split 0:2
    path = bucket / "data" / "chunk-000" / "episode_000002.parquet"
    df = pd.read_parquet(path)
    df["action"] = [row * 1.5 for row in df["action"]]
    df.to_parquet(path)
    _make_ds(bucket)  # episode 2 is outside the train split: stats stay valid


def test_unpinned_gripper_stats_are_rejected(bucket):
    payload = compute_yam_stats(bucket)
    payload[ACTION_STATS_KEY]["max"][13] = 0.99
    np.save(default_stats_path(bucket), payload, allow_pickle=True)
    with pytest.raises(ValueError, match="gripper min/max must be pinned"):
        _make_ds(bucket)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"unify_action_map": None}, "requires an explicit unify_action_map"),
        ({"action_mode": "eef"}, "action_mode='joint'"),
        ({"normalize_mode": "quantile"}, "min-max"),
    ],
)
def test_rejects_unsupported_settings(bucket, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make_ds(bucket, **kwargs)


def test_rejects_non_v21_layout(bucket):
    info_path = bucket / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v3.0"
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="LeRobot v2.1"):
        _make_ds(bucket, normalize_mode=None)
