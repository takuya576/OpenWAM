"""LeRobot v2.1 reader for bimanual I2RT YAM joint-space recordings.

YAM teleoperation exports store 14-D absolute joint positions per frame,

    [left_joint_1..6, left_gripper, right_joint_1..6, right_gripper]

for both ``action`` (leader-arm command) and ``observation.state`` (follower
position). Grippers use i2rt's normalized aperture: 0 = closed, 1 = open.

The policy is trained and served in joint space, so execution needs no IK.
Under ``unify_action`` the grippers keep their α slots (9 left, 43 right) and
the twelve arm joints take the twelve reserved slots that the α layout leaves
to embodiment-specific channels (left 68-73, right 74-79); the EEF pose slots
stay masked. Normalization is min-max over the train split (the α protocol)
from a stats file with two YAM-specific rules, applied by
:mod:`openwam.dataloader.utils.stats_computation.yam_stats_computation`:

* grippers use their physical [0, 1] range, so -1 = closed and +1 = open in
  every recording, as the α gripper slots require;
* an arm joint whose recorded span is below ``MIN_JOINT_SPAN_RAD`` (an arm
  held still for the whole task) gets that span around its midpoint, so
  sensor noise is not stretched across [-1, 1].

``delta_action_dims`` (openpi's ``DeltaActions``) trains the listed arm
joints relative to the window's first state, ``action[t + k] - state[t]``,
before normalization; the action stats are then fitted on those relative
chunks. Grippers stay absolute. Deploy adds the measured state back
(:class:`openwam.deploy.engine.JointInferenceEngine`), so the policy still
serves absolute joint targets.

The on-disk layout is LeRobot v2.1: one parquet and one mp4 per camera per
episode, with ``meta/episodes.jsonl`` and ``meta/tasks.jsonl``. Each episode
is presented to the base reader as its own v3 file, so window slicing, video
decoding and the multiview canvas are inherited unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.lerobotv3 import apply_info_splits
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.unify_action import parse_dim_list

logger = logging.getLogger(__name__)

ACTION_MODE = "joint"
ACTION_STATS_KEY = ACTION_MODE
STATE_STATS_KEY = f"{ACTION_MODE}_state"
JOINT_DIM = 14
JOINT_NAMES = [
    *(f"left_joint_{i}" for i in range(1, 7)),
    "left_gripper",
    *(f"right_joint_{i}" for i in range(1, 7)),
    "right_gripper",
]
GRIPPER_DIMS = (6, 13)
ARM_JOINT_DIMS = tuple(d for d in range(JOINT_DIM) if d not in GRIPPER_DIMS)
# i2rt maps the gripper motor range [closed, open] onto [0, 1].
GRIPPER_RANGE = (0.0, 1.0)
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
MIN_JOINT_SPAN_RAD = 0.1
NORMALIZATION_STATS_PREFIX = "yam_normalization_stats"

HEAD_CAMERA = "observation.images.cam_high"
LEFT_WRIST_CAMERA = "observation.images.cam_left_wrist"
RIGHT_WRIST_CAMERA = "observation.images.cam_right_wrist"

# A half-open "start:end" episode_index range, or an explicit list of episode indices.
SplitSpec = Union[str, List[int]]


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_delta_dims(spec: Any) -> Tuple[int, ...]:
    """Joint dims trained relative to the chunk-start state; ``None`` = all absolute.

    ``spec`` uses the ``unify_action_map`` range syntax, e.g. ``["0-5", "7-12"]``
    for every arm joint (openpi's YAM mask). Grippers must stay absolute: their
    α slots mean -1 = closed, +1 = open.
    """
    if spec is None:
        return ()
    dims = parse_dim_list(spec, JOINT_DIM)
    grippers = sorted(set(dims) & set(GRIPPER_DIMS))
    if grippers:
        raise ValueError(f"YAM delta_action_dims must leave the grippers {list(GRIPPER_DIMS)} absolute, got {grippers}")
    return dims


def _split_spec(spec: Any) -> SplitSpec:
    """A split as a half-open ``"start:end"`` range, or an explicit list of episode indices.

    Lists come back sorted, so a subset is named and fingerprinted the same however it is written.
    """
    if isinstance(spec, str):
        return spec
    episodes = sorted(int(ep) for ep in spec)
    if len(set(episodes)) != len(episodes):
        raise ValueError(f"YAM split episode list has duplicates: {episodes}")
    return episodes


def resolve_splits(dataset_dir: str | Path, splits: Optional[Mapping[str, SplitSpec]] = None) -> dict:
    """Episode splits: an explicit ``splits`` mapping overrides ``info.json``'s."""
    if splits is None:
        with open(Path(dataset_dir) / "meta" / "info.json", encoding="utf-8") as handle:
            splits = json.load(handle).get("splits", {}) or {}
    return {name: _split_spec(spec) for name, spec in splits.items()}


def select_split(
    eps: pd.DataFrame, split: str, split_map: Mapping[str, SplitSpec], *, source_name: str
) -> pd.DataFrame:
    """``apply_info_splits``, extended to a split given as an episode list (e.g. a per-position subset)."""
    spec = split_map.get(split)
    if not isinstance(spec, list):
        return apply_info_splits(eps, split, split_map, source_name=source_name)
    sel = eps[eps["episode_index"].isin(spec)].reset_index(drop=True)
    missing = sorted(set(spec) - set(sel["episode_index"].tolist()))
    if missing:
        raise ValueError(f"{source_name}: split {split} lists episodes {missing} that the dataset does not have")
    logger.info("%s: split=%s from an episode list (%d eps)", source_name, split, len(sel))
    return sel


def train_split_episodes(dataset_dir: str | Path, splits: Optional[Mapping[str, SplitSpec]] = None) -> pd.DataFrame:
    """``episode_index`` / ``length`` of the train-split episodes."""
    dataset_dir = Path(dataset_dir)
    eps = pd.DataFrame(read_jsonl(dataset_dir / "meta" / "episodes.jsonl"), columns=["episode_index", "length"])
    split_map = resolve_splits(dataset_dir, splits)
    return select_split(eps.astype(np.int64), "train", split_map, source_name=str(dataset_dir))


def train_episode_files(dataset_dir: str | Path, splits: Optional[Mapping[str, SplitSpec]] = None) -> List[Path]:
    """Parquet files of the train-split episodes, in episode order."""
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "meta" / "info.json", encoding="utf-8") as handle:
        info = json.load(handle)
    chunks_size = int(info.get("chunks_size", 1000))
    return [
        dataset_dir / info["data_path"].format(episode_chunk=ep // chunks_size, episode_index=ep)
        for ep in train_split_episodes(dataset_dir, splits)["episode_index"].tolist()
    ]


def train_data_fingerprint(dataset_dir: str | Path, splits: Optional[Mapping[str, SplitSpec]] = None) -> str:
    """SHA-256 over the train-split parquet files, so re-recorded data invalidates fitted stats."""
    digest = hashlib.sha256()
    for path in train_episode_files(dataset_dir, splits):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def default_stats_path(
    dataset_dir: str | Path,
    splits: Optional[Mapping[str, SplitSpec]] = None,
    *,
    delta_horizon: Optional[int] = None,
) -> Path:
    """Stats file of the train split, named by its range so a new split never reuses stale stats.

    An episode list is named by its size and a hash of its members: ``_train_14eps_<sha8>``.
    Delta-action stats depend on the chunk length too and add ``_delta<horizon>``.
    """
    train = resolve_splits(dataset_dir, splits).get("train")
    if isinstance(train, list):
        digest = hashlib.sha256(",".join(map(str, train)).encode()).hexdigest()[:8]
        suffix = f"_train_{len(train)}eps_{digest}"
    else:
        suffix = f"_train_{train.replace(':', '-')}" if train else ""
    if delta_horizon is not None:
        suffix += f"_delta{int(delta_horizon)}"
    return Path(dataset_dir) / "meta" / f"{NORMALIZATION_STATS_PREFIX}{suffix}.npy"


def _v3_path_template(template: str) -> str:
    """Rename v2.1 path placeholders to the v3 names the base reader formats with."""
    return template.replace("{episode_chunk", "{chunk_index").replace("{episode_index", "{file_index")


class YamDataset(LeRobotV3Reader):
    """Single-bucket bimanual YAM reader serving 14-D joint actions."""

    DATASET_NAME = "YAM"
    ACTION_DIM = JOINT_DIM
    NEEDED_COLS = ("action", "observation.state", "task_index")
    HEAD_CAMERA = HEAD_CAMERA
    LEFT_WRIST_CAMERA = LEFT_WRIST_CAMERA
    RIGHT_WRIST_CAMERA = RIGHT_WRIST_CAMERA
    DEPLOY_ACTION_MODE = ACTION_MODE
    DEFAULT_NORMALIZE_MODE = "min-max"
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + (
        "action_mode",
        "normalization_stats_path",
        "splits",
        "delta_action_dims",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        action_mode: str = ACTION_MODE,
        normalization_stats_path: Optional[str] = None,
        splits: Optional[Mapping[str, SplitSpec]] = None,
        delta_action_dims: Optional[Any] = None,
        unify_action: bool = False,
        unify_action_map: Optional[Any] = None,
        **kwargs: Any,
    ):
        if str(action_mode).strip().lower() != ACTION_MODE:
            raise ValueError(f"YAM supports only action_mode={ACTION_MODE!r}, got {action_mode!r}.")
        if bool(unify_action) and unify_action_map is None:
            # The identity map would scatter joint angles into the EEF pose slots.
            raise ValueError(
                "YAM unify_action=true requires an explicit unify_action_map; "
                'use ["68-73", 9, "74-79", 43] (grippers -> α gripper slots, joints -> reserved slots)'
            )
        self.action_mode = ACTION_MODE
        self._delta_dims = resolve_delta_dims(delta_action_dims)
        self._splits = dict(splits) if splits is not None else None
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._state_normalization_stats: Optional[dict] = None
        super().__init__(
            dataset_dir=dataset_dir,
            unify_action=bool(unify_action),
            unify_action_map=unify_action_map,
            **kwargs,
        )

    def _resolve_cameras(self, info: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        # An explicit layout is the canvas order [head, left, right]; decode
        # exactly those streams so the yaml fully controls which view goes where.
        if self._multiview and self._camera_layout_param is not None:
            layout = list(self._camera_layout_param)[:3]
            layout += [None] * (3 - len(layout))
            return tuple(str(camera) if camera else None for camera in layout)
        return super()._resolve_cameras(info)

    def _build_episode_index(self, info: dict) -> pd.DataFrame:
        version = info.get("codebase_version")
        if version != "v2.1":
            raise ValueError(f"YAM reads LeRobot v2.1 datasets, got codebase_version={version!r}")
        self._data_path_template = _v3_path_template(info["data_path"])
        self._video_path_template = _v3_path_template(info["video_path"])

        eps = pd.DataFrame(
            read_jsonl(self._dataset_dir / "meta" / "episodes.jsonl"), columns=["episode_index", "length"]
        )
        eps = eps.astype(np.int64)
        chunk = eps["episode_index"] // int(info.get("chunks_size", 1000))
        eps["data/chunk_index"] = chunk
        eps["data/file_index"] = eps["episode_index"]
        eps["_data_row_offset"] = 0
        for camera in self._video_cameras():
            eps[f"videos/{camera}/chunk_index"] = chunk
            eps[f"videos/{camera}/file_index"] = eps["episode_index"]
            eps[self._video_offset_col(camera)] = 0
        split_map = resolve_splits(self._dataset_dir, self._splits)
        return select_split(eps, self._split, split_map, source_name=f"YAM({self._dataset_id})")

    def _load_prompts(self) -> None:
        rows = read_jsonl(self._dataset_dir / "meta" / "tasks.jsonl")
        self._task_idx_to_text = {int(row["task_index"]): str(row["task"]) for row in rows}

    def _post_init(self, info: dict) -> None:
        if self._multiview and (self._height, self._width) != (384, 320):
            raise ValueError(f"YAM multiview requires height=384, width=320, got {self._height}x{self._width}")
        features = info.get("features", {}) or {}
        for column in ("action", "observation.state"):
            feature = features.get(column, {})
            shape = tuple(feature.get("shape", ()))
            if shape != (JOINT_DIM,):
                raise ValueError(f"YAM {column} feature must have shape [{JOINT_DIM}], got {shape}")
            names = feature.get("names")
            if names is not None and list(names) != JOINT_NAMES:
                raise ValueError(f"YAM {column} names must be {JOINT_NAMES}, got {names}")
        for camera in (self._head_camera, self._left_wrist_camera, self._right_wrist_camera):
            if camera is not None and features.get(camera, {}).get("dtype") != "video":
                raise ValueError(f"YAM camera {camera!r} is not a video feature of {self._dataset_dir}")

    @property
    def _delta_horizon(self) -> Optional[int]:
        """Action steps per window, the span delta stats are fitted over; None for absolute actions."""
        return self._num_frames - 1 if self._delta_dims else None

    def _build_stats_rank0(self, path: Path) -> None:
        """Auto-build the stats file: rank 0 scans, other ranks wait."""
        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        if dist_ready:
            rank = dist.get_rank()
        else:
            # torchrun sets RANK before init_process_group; honor it so
            # pre-init constructions still elect a single builder.
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            from openwam.dataloader.utils.stats_computation.yam_stats_computation import build_and_save_yam_stats

            logger.info("No YAM stats at %s — scanning the train split (rank 0; other ranks wait)", path)
            build_and_save_yam_stats(
                self._dataset_dir,
                output=path,
                splits=self._splits,
                delta_action_dims=self._delta_dims or None,
                delta_horizon=self._delta_horizon,
            )
        else:
            deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
            poll_interval_s = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
            while not path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for rank 0 to build YAM stats: {path}")
                time.sleep(poll_interval_s)

    def _load_stats(self, info: dict) -> Optional[dict]:
        if self._normalize_mode in (None, "none", "null"):
            self._state_normalization_stats = None
            return None
        if self._normalize_mode != "min-max":
            raise ValueError(
                f"YAM supports normalize_mode min-max (the α protocol) or null, got {self._normalize_mode!r}"
            )
        stats_path = (
            Path(self._source_stats_path)
            if self._source_stats_path
            else default_stats_path(self._dataset_dir, self._splits, delta_horizon=self._delta_horizon)
        )
        if not stats_path.is_file():
            self._build_stats_rank0(stats_path)

        raw = np.load(stats_path, allow_pickle=True).item()
        if not isinstance(raw, dict) or ACTION_STATS_KEY not in raw or STATE_STATS_KEY not in raw:
            raise KeyError(f"{stats_path} must contain {ACTION_STATS_KEY!r} and {STATE_STATS_KEY!r} blocks")
        # Training normalizes from these blocks and deploy unnormalizes from the
        # same file, so reject a mismatched file rather than patching it here.
        train_split = resolve_splits(self._dataset_dir, self._splits).get("train", "all")
        fingerprint = train_data_fingerprint(self._dataset_dir, self._splits)
        delta = {"delta_action_dims": list(self._delta_dims), "delta_horizon": self._delta_horizon}
        # Files from before delta actions carry neither key: absolute actions.
        fitted = {
            "delta_action_dims": [int(d) for d in raw[ACTION_STATS_KEY].get("delta_action_dims", [])],
            "delta_horizon": raw[ACTION_STATS_KEY].get("delta_horizon"),
        }
        if fitted != delta:
            raise ValueError(
                f"{stats_path}:{ACTION_STATS_KEY} was fitted on actions with {fitted}, but this run trains with "
                f"{delta}; delete the file to rebuild it."
            )
        materialized = []
        for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
            block = raw[key]
            if block.get("gripper_convention") != GRIPPER_CONVENTION:
                raise ValueError(
                    f"{stats_path}:{key} declares gripper_convention={block.get('gripper_convention')!r}, "
                    f"expected {GRIPPER_CONVENTION!r}"
                )
            if block.get("train_split") != train_split:
                raise ValueError(
                    f"{stats_path}:{key} was fitted on train split {block.get('train_split')!r}, "
                    f"but this run trains on {train_split!r}"
                )
            if block.get("source_sha256") != fingerprint:
                raise ValueError(
                    f"{stats_path}:{key} was fitted on different train-split data than is on disk now; "
                    "delete the file to rebuild it."
                )
            stats = materialize_eef_stats(
                dict(block), "min-max", dim=JOINT_DIM, strict_minmax=True, source_hint=f"{stats_path}:{key}"
            )
            grip = list(GRIPPER_DIMS)
            if not (
                np.allclose(stats["min"][grip], GRIPPER_RANGE[0]) and np.allclose(stats["max"][grip], GRIPPER_RANGE[1])
            ):
                raise ValueError(f"{stats_path}:{key} gripper min/max must be pinned to {GRIPPER_RANGE}")
            materialized.append(stats)
        action_stats, self._state_normalization_stats = materialized
        self.normalization_stats_path = str(stats_path)
        return action_stats

    @staticmethod
    def _read_joint_column(win: pd.DataFrame, column: str) -> np.ndarray:
        values = np.stack(win[column].values).astype(np.float32)
        if values.ndim != 2 or values.shape[1] != JOINT_DIM:
            raise ValueError(f"YAM {column} must be (T, {JOINT_DIM}), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"YAM {column} contains NaN or infinity")
        return values

    def _action_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        actions = self._read_joint_column(win, "action")
        if self._delta_dims:
            dims = list(self._delta_dims)
            actions[:, dims] -= self._read_joint_column(win, "observation.state")[0, dims]
        return apply_normalization(actions, self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        state = self._read_joint_column(win, "observation.state")[0:1]
        return apply_normalization(state, self._state_normalization_stats, self._normalize_mode)


__all__ = [
    "ACTION_MODE",
    "ACTION_STATS_KEY",
    "ARM_JOINT_DIMS",
    "GRIPPER_CONVENTION",
    "GRIPPER_DIMS",
    "GRIPPER_RANGE",
    "JOINT_DIM",
    "JOINT_NAMES",
    "MIN_JOINT_SPAN_RAD",
    "NORMALIZATION_STATS_PREFIX",
    "STATE_STATS_KEY",
    "YamDataset",
    "default_stats_path",
    "read_jsonl",
    "resolve_delta_dims",
    "resolve_splits",
    "select_split",
    "train_data_fingerprint",
    "train_episode_files",
    "train_split_episodes",
]
