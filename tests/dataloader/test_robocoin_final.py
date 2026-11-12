"""Regression tests for the physically cleaned RoboCOIN training contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from openwam.dataloader.robocoin import MultiRobotCOINDataset, RoboCOINDataset
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20

CAMERA = "observation.images.cam_head_rgb"


def _make_bucket(bucket: Path, lengths=(8, 8), *, robot_type="test_robot", dex=False):
    meta = bucket / "meta"
    (meta / "episodes").mkdir(parents=True)
    features = {
        CAMERA: {"dtype": "video"},
        "eef_sim_pose_action": {},
        "eef_sim_pose_state": {},
    }
    if dex:
        motors = ["left_hand_joint_0", "left_hand_joint_1", "right_hand_joint_0"]
        features.update({"action": {"names": motors}, "observation.state": {"names": motors}})
    else:
        features.update({"gripper_open_scale_action": {}, "gripper_open_scale_state": {}})
    info = {
        "fps": 30,
        "robot_type": robot_type,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    starts = np.cumsum([0, *lengths[:-1]])
    pd.DataFrame(
        {
            "episode_index": range(len(lengths)),
            "length": lengths,
            "dataset_from_index": starts,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{CAMERA}/chunk_index": 0,
            f"videos/{CAMERA}/file_index": 0,
        }
    ).to_parquet(meta / "episodes" / "chunk-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["move the object"], name="task")).to_parquet(
        meta / "tasks.parquet"
    )
    n = sum(lengths)
    pose = np.zeros((n, 12), dtype=np.float32)
    pose[:, 0] = np.arange(n)
    data = {
        "task_index": np.zeros(n, dtype=np.int64),
        "episode_index": np.repeat(np.arange(len(lengths)), lengths),
        "frame_index": np.concatenate([np.arange(length) for length in lengths]),
        "eef_sim_pose_action": list(pose),
        "eef_sim_pose_state": list(pose),
    }
    if dex:
        fingers = np.tile([0.1, 0.2, 0.3], (n, 1)).astype(np.float32)
        data.update({"action": list(fingers), "observation.state": list(fingers / 2)})
    else:
        grip = np.tile([0.25, 0.75], (n, 1)).astype(np.float32)
        data.update({"gripper_open_scale_action": list(grip), "gripper_open_scale_state": list(grip)})
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pd.DataFrame(data).to_parquet(data_dir / "file-000.parquet", index=False)
    return bucket


def _vectors(dim):
    return {
        "mean": [0.0] * dim,
        "std": [1.0] * dim,
        "min": [-1.0] * dim,
        "max": [1.0] * dim,
        "q01": [-1.0] * dim,
        "q99": [1.0] * dim,
    }


def _write_stats(family, *, robot_type="test_robot", dex=False, payload=None):
    meta = family / "meta"
    meta.mkdir(exist_ok=True)
    raw = payload if payload is not None else {"eef": {**_vectors(20), "robot_type": robot_type}}
    if dex:
        raw["hand"] = {**_vectors(3), "dof_left": 2, "dof_right": 1}
    path = meta / f"stats_{robot_type}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture
def decode_frame_indices(monkeypatch):
    def decode(path, frame_indices, height, width):
        del path
        return [Image.new("RGB", (width, height), (int(index), 0, 0)) for index in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", decode)


def test_family_and_single_bucket_read_existing_stats_after_relocation(tmp_path):
    original = tmp_path / "old_family_name"
    _make_bucket(original / "bucket-a")
    _make_bucket(original / "bucket-b")
    path = _write_stats(original)
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Historical metadata is informational; no old filesystem path is required.
    raw.update({"trim_provenance": {"path": "/retired/audit.csv"}, "whole_bucket_exclusions_provenance": {}})
    path.write_text(json.dumps(raw), encoding="utf-8")
    family = tmp_path / "relocated_RoboCOIN"
    original.rename(family)

    dataset = RoboCOINDataset.from_config({"dataset_dir": str(family), "normalize_mode": "quantile"})
    assert isinstance(dataset, MultiRobotCOINDataset)
    assert len(dataset._buckets) == 2
    assert sum(len(bucket._eps_df) for bucket in dataset._buckets) == 4
    for bucket in dataset._buckets:
        np.testing.assert_array_equal(bucket._normalization_stats["q99"], np.ones(20))
    leaf = RoboCOINDataset.from_config({"dataset_dir": str(family / "bucket-a"), "normalize_mode": "quantile"})
    assert len(leaf._eps_df) == 2


def test_processed_manifest_is_used_without_additional_exclusions(tmp_path):
    family = tmp_path / "family"
    former_denied_name = "Airbot_MMK2_storage_peach_pear"
    bucket = _make_bucket(family / former_denied_name, lengths=(8, 5, 1))
    # This obsolete artifact must never be opened by a final-data reader.
    (bucket / "meta" / "excluded_episodes.json").write_text("invalid old artifact", encoding="utf-8")
    dataset = RoboCOINDataset.from_config({"dataset_dir": str(family), "normalize_mode": None})
    leaf = dataset._buckets[0]
    assert leaf._eps_df["episode_index"].tolist() == [0, 1, 2]
    assert leaf._eps_df["length"].tolist() == [8, 5, 1]
    assert leaf._ep_data_row_offset.tolist() == [0, 8, 13]
    assert len(leaf) == 14
    assert "trim_csv" not in leaf.CONFIG_KEYS


def test_missing_stats_fail_without_writing_or_computing(tmp_path):
    bucket = _make_bucket(tmp_path / "family" / "bucket")
    files_before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(FileNotFoundError, match="stats_test_robot.json"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode="quantile")
    assert files_before == {
        path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("bad_value", [[0.0] * 19, [float("nan")] * 20])
def test_invalid_stored_stat_vectors_fail_early(tmp_path, bad_value):
    family = tmp_path / "family"
    bucket = _make_bucket(family / "bucket")
    raw = {"eef": _vectors(20)}
    raw["eef"]["q01"] = bad_value
    _write_stats(family, payload=raw)
    with pytest.raises(ValueError, match="finite vector with shape"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode="quantile")


def test_stats_robot_type_must_match_bucket(tmp_path):
    family = tmp_path / "family"
    bucket = _make_bucket(family / "bucket")
    _write_stats(family, payload={"eef": {**_vectors(20), "robot_type": "other_robot"}})
    with pytest.raises(ValueError, match="robot_type"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode="quantile")


def test_active_normalization_fields_are_required(tmp_path):
    family = tmp_path / "family"
    bucket = _make_bucket(family / "bucket")
    _write_stats(family, payload={"eef": {"min": [-1.0] * 20, "max": [1.0] * 20}})
    with pytest.raises(KeyError, match="q01"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode="quantile")


def test_split_keeps_physical_data_and_video_offsets_aligned(tmp_path, decode_frame_indices):
    bucket = _make_bucket(tmp_path / "bucket")
    path = bucket / "meta" / "info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["splits"] = {"train": "1:2"}
    path.write_text(json.dumps(info), encoding="utf-8")
    dataset = RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None, num_frames=5, video_stride=1)
    assert dataset._ep_data_row_offset.tolist() == [8]
    assert dataset._ep_video_frame_offsets[CAMERA].tolist() == [8]
    observed = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        action_rows = sample["action"][sample["action_mask"].any(dim=1), 0].to(torch.int64).tolist()
        video_rows = [image.getpixel((0, 0))[0] for image, valid in zip(sample["video"], sample["video_mask"]) if valid]
        assert action_rows == video_rows[: len(action_rows)]
        observed.update(video_rows)
        np.testing.assert_allclose(sample["action"][0, [9, 19]], [0.25, 0.75])
    assert observed == set(range(8, 16))


def test_rot6d_remains_identity_under_stored_normalization(tmp_path):
    family = tmp_path / "family"
    bucket = _make_bucket(family / "bucket")
    raw = {"eef": _vectors(20)}
    raw["eef"]["q01"] = [-2.0] * 20
    raw["eef"]["q99"] = [2.0] * 20
    _write_stats(family, payload=raw)
    dataset = RoboCOINDataset(dataset_dir=str(bucket), normalize_mode="quantile")
    np.testing.assert_array_equal(dataset._normalization_stats["q01"][list(ROT6D_DIMS_EEF20)], -1)
    np.testing.assert_array_equal(dataset._normalization_stats["q99"][list(ROT6D_DIMS_EEF20)], 1)
    assert dataset._normalization_stats["q01"][0] == -2


def test_dexterous_hand_keeps_fingers_and_masks_scalar_grips(tmp_path, decode_frame_indices):
    family = tmp_path / "family"
    bucket = _make_bucket(family / "dex", dex=True)
    _write_stats(family, dex=True)
    dataset = RoboCOINDataset(
        dataset_dir=str(bucket), normalize_mode="quantile", unify_action=True, num_frames=5, video_stride=1
    )
    sample = dataset[0]
    assert sample["action"].shape[-1] == 80
    np.testing.assert_allclose(sample["action"][0, [10, 11, 44]], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(sample["proprio"][0, [10, 11, 44]], [0.05, 0.1, 0.15], atol=1e-6)
    assert sample["action_mask"][0, [10, 11, 44]].all()
    assert not sample["action_mask"][0, [9, 43, 12, 45, 79]].any()


def test_dexterous_hand_stats_must_match_finger_dof(tmp_path):
    family = tmp_path / "family"
    bucket = _make_bucket(family / "dex", dex=True)
    _write_stats(family, payload={"eef": _vectors(20), "hand": {**_vectors(3), "dof_left": 1, "dof_right": 2}})
    with pytest.raises(ValueError, match="DOF"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode="quantile", unify_action=True)


def test_video_timestamps_are_independent_of_contiguous_data_rows(tmp_path, decode_frame_indices):
    """A native video shard may contain padding absent from the data shard."""
    bucket = _make_bucket(tmp_path / "bucket")
    manifest = bucket / "meta" / "episodes" / "chunk-000.parquet"
    episodes = pd.read_parquet(manifest)
    episodes[f"videos/{CAMERA}/from_timestamp"] = np.array([100, 150]) / 30.0
    episodes[f"videos/{CAMERA}/to_timestamp"] = np.array([108, 158]) / 30.0
    episodes.to_parquet(manifest, index=False)
    dataset = RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None, num_frames=5, video_stride=1)
    first = dataset._getitem_impl(0)
    second = dataset._getitem_impl(int(dataset._cum_n_starts[1]))
    assert first["action"][0, 0].item() == 0
    assert second["action"][0, 0].item() == 8
    assert first["video"][0].getpixel((0, 0))[0] == 100
    assert second["video"][0].getpixel((0, 0))[0] == 150


@pytest.mark.parametrize("bad_start", [-1.0, float("nan"), 0.05])
def test_invalid_video_timestamps_fail_before_sampling(tmp_path, bad_start):
    bucket = _make_bucket(tmp_path / "bucket")
    manifest = bucket / "meta" / "episodes" / "chunk-000.parquet"
    episodes = pd.read_parquet(manifest)
    episodes[f"videos/{CAMERA}/from_timestamp"] = [bad_start, 8 / 30.0]
    episodes.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="timestamp|frame boundar"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None)


def test_family_stats_cannot_fall_back_outside_selected_root(tmp_path):
    family = tmp_path / "family"
    _make_bucket(family / "bucket")
    _write_stats(tmp_path)
    with pytest.raises(FileNotFoundError, match="stats_test_robot.json"):
        RoboCOINDataset.from_config({"dataset_dir": str(family), "normalize_mode": "quantile"})


def test_float32_video_timestamps_keep_long_shard_frame_offsets(tmp_path):
    bucket = _make_bucket(tmp_path / "bucket")
    manifest = bucket / "meta" / "episodes" / "chunk-000.parquet"
    episodes = pd.read_parquet(manifest)
    episodes[f"videos/{CAMERA}/from_timestamp"] = np.array([300000, 300008], dtype=np.float64).astype(
        np.float32
    ) / np.float32(30)
    episodes[f"videos/{CAMERA}/to_timestamp"] = np.array([300008, 300016], dtype=np.float64).astype(
        np.float32
    ) / np.float32(30)
    episodes.to_parquet(manifest, index=False)
    dataset = RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None)
    assert dataset._ep_video_frame_offsets[CAMERA].tolist() == [300000, 300008]


def test_video_timestamp_span_must_match_episode_length(tmp_path):
    bucket = _make_bucket(tmp_path / "bucket")
    manifest = bucket / "meta" / "episodes" / "chunk-000.parquet"
    episodes = pd.read_parquet(manifest)
    episodes[f"videos/{CAMERA}/from_timestamp"] = np.array([0, 8]) / 30.0
    episodes[f"videos/{CAMERA}/to_timestamp"] = np.array([9, 16]) / 30.0
    episodes.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="span.*episode length"):
        RoboCOINDataset(dataset_dir=str(bucket), normalize_mode=None)
