"""DROID clean-export loading uses shipped stats without rescanning source rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from openwam.dataloader.oxe_droid import (
    DROID_EEF_STATS_CONTRACT,
    DROID_EEF_STATS_CONTRACT_KEY,
    OxeDroidDataset,
)
from openwam.dataloader.utils.lerobotv3 import DataContractError


@pytest.fixture
def droid_export(tmp_path, monkeypatch):
    root = tmp_path / "Droid"
    meta = root / "meta"
    (meta / "episodes").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 15,
                "total_episodes": 2,
                "total_frames": 48,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": {},
            }
        )
    )
    rows = []
    for episode in range(2):
        row = {
            "episode_index": episode,
            "length": 24,
            "dataset_from_index": 24 * episode,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for camera in ("observation.images.primary", "observation.images.wrist"):
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = 0
        rows.append(row)
    pd.DataFrame(rows).to_parquet(meta / "episodes" / "chunk-000.parquet")
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["pick up the cup", "place the cup"], name="task")).to_parquet(
        meta / "tasks.parquet"
    )
    action = np.zeros((48, 7), dtype=np.float32)
    action[:, 0] = np.arange(48) / 100
    action[:, 6] = 0.25
    pose = np.zeros((48, 6), dtype=np.float32)
    pose[:, 0] = 0.75
    state = np.zeros((48, 7), dtype=np.float32)
    state[:, :3] = 100  # TCP values must never replace the rigid mount pose.
    state[:, 6] = 0.8
    pd.DataFrame(
        {
            "episode_index": np.repeat([0, 1], 24),
            "frame_index": np.tile(np.arange(24), 2),
            "task_index": np.repeat([0, 1], 24),
            "state": list(state),
            "other_information.observation_gripper_pose6d": list(pose),
            "other_information.action_wrist_pose": list(action),
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    stats = {
        "mean": [0.0] * 10,
        "std": [1.0] * 10,
        "min": [-2.0] * 10,
        "max": [2.0] * 10,
        "q01": [-2.0] * 10,
        "q99": [2.0] * 10,
        DROID_EEF_STATS_CONTRACT_KEY: DROID_EEF_STATS_CONTRACT,
    }
    (meta / "eef_stats.json").write_text(json.dumps(stats))
    monkeypatch.setattr(
        "openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames",
        lambda path, frame_indices, height, width: [Image.new("RGB", (width, height)) for _ in frame_indices],
    )
    return root


def _reader(root, **kwargs):
    return OxeDroidDataset(str(root), num_frames=9, video_stride=4, height=32, width=32, **kwargs)


def test_clean_droid_loads_without_exclusion_certificate_or_fallback_columns(droid_export):
    ds = _reader(droid_export, normalize_mode=None)
    assert ds._eps_df["episode_index"].tolist() == [0, 1]
    sample = ds[0]
    assert sample["prompt"] == "pick up the cup"
    assert sample["action"].shape == (8, 20)
    assert sample["proprio"].shape == (1, 20)
    assert sample["action_mask"][:, :10].all()
    assert not sample["action_mask"][:, 10:].any()
    assert sample["proprio"][0, 0].item() == pytest.approx(0.75)
    assert sample["action"][0, 9].item() == pytest.approx(0.75)
    assert sample["proprio"][0, 9].item() == pytest.approx(0.2)
    np.testing.assert_allclose(sample["action"][0, 3:9], [1, 0, 0, 0, 1, 0])
    assert ds[int(ds._cum_n_starts[1])]["prompt"] == "place the cup"


def test_droid_shipped_stats_are_used_without_mutating_export(droid_export):
    before = {str(p.relative_to(droid_export)): p.stat().st_mtime_ns for p in droid_export.rglob("*") if p.is_file()}
    ds = _reader(droid_export)
    sample = ds[0]
    assert sample["action"][0, 9].item() == pytest.approx(0.375)
    np.testing.assert_allclose(sample["action"][0, 3:9], [1, 0, 0, 0, 1, 0])
    after = {str(p.relative_to(droid_export)): p.stat().st_mtime_ns for p in droid_export.rglob("*") if p.is_file()}
    assert before == after


def test_droid_startup_does_not_scan_numeric_or_prompt_source_shards(droid_export, monkeypatch):
    read_parquet = pd.read_parquet

    def metadata_only(path, *args, **kwargs):
        assert "data" not in Path(path).relative_to(droid_export).parts
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", metadata_only)
    assert len(_reader(droid_export)) == 48


def test_legacy_external_exclusions_cannot_reapply_filtering(droid_export):
    (droid_export / "meta" / "excluded_episodes.json").write_text("not a JSON artifact")
    assert _reader(droid_export)._eps_df["episode_index"].tolist() == [0, 1]


def test_missing_droid_stats_fail_without_creating_output(droid_export):
    path = droid_export / "meta" / "eef_stats.json"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="eef_stats.json"):
        _reader(droid_export)
    assert not path.exists()


def test_droid_stats_reject_incompatible_pose_contract(droid_export):
    path = droid_export / "meta" / "eef_stats.json"
    stats = json.loads(path.read_text())
    stats[DROID_EEF_STATS_CONTRACT_KEY]["action_pose_source"] = "other_information.action_tcp_pose"
    path.write_text(json.dumps(stats))
    with pytest.raises(DataContractError, match="terminal-arm"):
        _reader(droid_export)


def test_droid_stats_reject_wrong_width(droid_export):
    path = droid_export / "meta" / "eef_stats.json"
    stats = json.loads(path.read_text())
    stats["mean"] = [0.0] * 11
    path.write_text(json.dumps(stats))
    with pytest.raises((DataContractError, ValueError), match="10|shape"):
        _reader(droid_export)


def test_droid_cannot_borrow_another_datasets_parent_statistics(droid_export):
    local = droid_export / "meta" / "eef_stats.json"
    outer = droid_export.parent / "meta" / "eef_stats.json"
    outer.parent.mkdir()
    outer.write_bytes(local.read_bytes())
    local.unlink()
    with pytest.raises(FileNotFoundError, match="DROID requires its own"):
        _reader(droid_export)
