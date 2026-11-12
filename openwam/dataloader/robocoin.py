"""Read physically cleaned RoboCOIN LeRobot v3 task buckets.

Action and state use the published Euler-XYZ EEF poses and gripper openness
(``0=closed, 1=open``). Dexterous-hand buckets retain their finger joints under
unified actions. Normalization reads the existing per-robot statistics from
``meta/stats_<robot_type>.json`` under the supplied family root; loading never
changes the dataset or applies an additional quality filter.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import EEF_DIM as _ACTION_DIM
from openwam.dataloader.utils.eef import eef14_to_eef20
from openwam.dataloader.utils.normalization import (
    STAT_KEYS,
    apply_normalization,
    find_dataset_stats,
    materialize_eef_stats,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _read_robot_stats(path: str) -> dict:
    """Share immutable on-disk stats across buckets of the same robot type."""
    with open(path, encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"RoboCOIN stats {path} must contain a JSON object")
    return raw


def _materialize_robot_stats(raw, *, dim: int, mode: str, source: str, rot6d: bool = False) -> dict:
    """Validate stored vectors before applying the normalization contract."""
    if not isinstance(raw, dict):
        raise ValueError(f"RoboCOIN stats {source} must contain a mapping")
    for key in STAT_KEYS:
        if key not in raw:
            continue
        values = np.asarray(raw[key], dtype=np.float32)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(
                f"RoboCOIN stats {source}.{key} must be a finite vector with shape ({dim},), got {values.shape}"
            )
    return materialize_eef_stats(
        raw, mode, dim=dim, strict_minmax=False, source_hint=source, force_rot6d_identity=rot6d
    )


_NEEDED_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "gripper_open_scale_action",
    "eef_sim_pose_state",
    "gripper_open_scale_state",
)


_BASE_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "eef_sim_pose_state",
)
_GRIP_COLS = ("gripper_open_scale_action", "gripper_open_scale_state")


_DEX_UNIFY_COLS = (
    "task_index",
    "eef_sim_pose_action",
    "eef_sim_pose_state",
    "action",
    "observation.state",
)


GRIP_EXCLUDED_DIM_MASK = np.ones(_ACTION_DIM, dtype=bool)
GRIP_EXCLUDED_DIM_MASK[9] = False
GRIP_EXCLUDED_DIM_MASK[19] = False


def _finger_indices(feature: dict):
    """Indices of left/right dexterous-hand joints within an action/state array.

    Reads the ``names`` (or ``names.motors``) list from one info.features entry
    and returns ``(left_idx, right_idx)`` for ``*_hand_joint_*`` motors.
    """

    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("motors")
    names = names or []
    left = [i for i, x in enumerate(names) if str(x).startswith("left_hand_joint")]
    right = [i for i, x in enumerate(names) if str(x).startswith("right_hand_joint")]
    return left, right


MAX_HAND_DOF = 24


def dex_finger_layout(features: dict):
    """Validated dexterous-hand finger layout for one ``info.features`` dict.

    A bucket qualifies as dexterous-hand ONLY when it has the unified pose column
    but NO ``gripper_open_scale_*`` columns, AND both hands carry
    ``1..MAX_HAND_DOF`` ``*_hand_joint_*`` motors with the action and state finger
    counts agreeing. Returns ``(aL, aR, sL, sR)`` — the left/right finger indices
    within the raw ``action`` / ``observation.state`` arrays — when all hold, else
    ``None``.

    Single source of truth for the dex-bucket gate, shared by the reader
    (:meth:`RoboCOINDataset.__init__`) and all callers so their hand layouts agree.
    """

    has_grip = all(c in features for c in _GRIP_COLS)
    if has_grip or "eef_sim_pose_action" not in features:
        return None
    aL, aR = _finger_indices(features.get("action", {}))
    sL, sR = _finger_indices(features.get("observation.state", {}))
    kL, kR = len(aL), len(aR)
    if not (0 < kL <= MAX_HAND_DOF and 0 < kR <= MAX_HAND_DOF and len(sL) == kL and len(sR) == kR):
        return None
    return aL, aR, sL, sR


def _build_dex_unify_map(k_left: int, k_right: int):
    """``dst_index`` (single-list spec) for a dex-hand raw action laid out as
    ``[L_pos(3), L_rot6d(6), L_fingers(kL), R_pos(3), R_rot6d(6), R_fingers(kR)]``.

    Maps pose → shared 80-D pose slots [0:9)/[R_POS:R_POS+9) and fingers → the
    dedicated hand slots [10:10+kL)/[R_HAND:R_HAND+kR) (front-aligned; the unused
    tail of each MAX_HAND_DOF-wide hand region and the gripper slots stay unmapped
    → masked). All right-side offsets are derived from MAX_HAND_DOF so the layout
    follows the hand width automatically (currently: R pose 34:43, R hand 44:68).
    """

    l_hand = 10
    r_pos = l_hand + MAX_HAND_DOF
    r_hand = r_pos + 10
    return (
        list(range(0, 3))
        + list(range(3, 9))
        + list(range(l_hand, l_hand + k_left))
        + list(range(r_pos, r_pos + 3))
        + list(range(r_pos + 3, r_pos + 9))
        + list(range(r_hand, r_hand + k_right))
    )


HEAD_CAMERA_PRIORITY = [
    "observation.images.cam_high_rgb",
    "observation.images.cam_head_rgb",
    "observation.images.cam_head_right_rgb",
    "observation.images.cam_head_left_rgb",
    "observation.images.cam_high_right_rgb",
    "observation.images.cam_high_left_rgb",
    "observation.images.cam_high_realsense_rgb",
    "observation.images.cam_front_rgb",
    "observation.images.cam_front_head_rgb",
    "observation.images.cam_front_chest_rgb",
    "observation.images.cam_chest_rgb",
    "observation.images.camera_head_rgb",
    "observation.images.cam_left_high",
    "observation.images.ego_view",
]

WRIST_LEFT_CANDIDATES = [
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_left_wrist_rgb_rgb",
    "observation.images.camera_left_wrist_rgb",
    "observation.images.cam_left_wrist",
]

WRIST_RIGHT_CANDIDATES = [
    "observation.images.cam_right_wrist_rgb",
    "observation.images.cam_right_wrist_rgb_rgb",
    "observation.images.camera_right_wrist_rgb",
    "observation.images.cam_right_wrist",
]


def _resolve_robocoin_cameras(features: dict) -> tuple:
    """Resolve head / left-wrist / right-wrist camera keys from info.json features.

    Returns:
        (head_cam, left_wrist_cam, right_wrist_cam) — each is a string or None.
    """

    feat_keys = set(features.keys())
    head = None
    for c in HEAD_CAMERA_PRIORITY:
        if c in feat_keys:
            head = c
            break
    left_wrist = None
    for c in WRIST_LEFT_CANDIDATES:
        if c in feat_keys:
            left_wrist = c
            break
    right_wrist = None
    for c in WRIST_RIGHT_CANDIDATES:
        if c in feat_keys:
            right_wrist = c
            break
    return head, left_wrist, right_wrist


class RoboCOINDataset(LeRobotV3Reader):
    """Single-dataset reader for one RoboCOIN task bucket (LeRobot v3).

    Bimanual 20-D EEF with real action/state supervision and per-robot-type
    in-reader normalization. All window / offset / video / pickle machinery is
    inherited from :class:`LeRobotV3Reader`; only RoboCOIN-specific bits are
    overridden below.
    """

    STRICT_BUCKET_LOADING = True
    DATASET_NAME = "RoboCOIN"
    NEEDED_COLS = _NEEDED_COLS

    PROMPT_FILE_REQUIRED = False

    WRIST_DECODE_TOLERATED = (Exception,)

    def __init__(
        self,
        dataset_dir,
        *,
        unify_action: bool = False,
        unify_action_map=None,
        **kwargs,
    ):
        """Detect dexterous-hand + unify buckets and wire finger encoding.

        Default (unify off, or grippered bucket): byte-identical to the inherited
        behavior. When ``unify_action=True`` AND this bucket is dexterous-hand
        (``eef_sim_pose_*`` present but no ``gripper_open_scale_*``), the reader
        emits a wider raw action ``[pose(18) + fingers(kL+kR)]`` and a per-bucket
        ``unify_action_map`` that scatters fingers into the dedicated 80-D hand
        slots [10:34)/[44:68) (overriding any config map). Requires unify so the
        variable-width raw collapses to the uniform 80-D head.
        """

        self._dex_unify = False
        self._k_left = 0
        self._k_right = 0
        if unify_action:
            try:
                with open(Path(dataset_dir) / "meta" / "info.json") as f:
                    features = json.load(f).get("features", {})
            except (OSError, ValueError):
                features = {}
            layout = dex_finger_layout(features)
            if layout is not None:
                aL, aR, sL, sR = layout
                kL, kR = len(aL), len(aR)
                self._dex_unify = True
                self._k_left, self._k_right = kL, kR
                self._fidx_act = (np.asarray(aL, dtype=np.int64), np.asarray(aR, dtype=np.int64))
                self._fidx_state = (np.asarray(sL, dtype=np.int64), np.asarray(sR, dtype=np.int64))

                self.ACTION_DIM = 18 + kL + kR
                unify_action_map = _build_dex_unify_map(kL, kR)
            elif "eef_sim_pose_action" in features and not all(c in features for c in _GRIP_COLS):
                aL, aR = _finger_indices(features.get("action", {}))
                sL, sR = _finger_indices(features.get("observation.state", {}))
                logger.warning(
                    "RoboCOIN %s: dexterous-hand bucket (pose, no gripper) but finger layout "
                    "failed the gate (action L/R=%d/%d, state L/R=%d/%d, max=%d); falling back "
                    "to pose-only under unify_action (fingers dropped).",
                    dataset_dir,
                    len(aL),
                    len(aR),
                    len(sL),
                    len(sR),
                    MAX_HAND_DOF,
                )
        super().__init__(dataset_dir, unify_action=unify_action, unify_action_map=unify_action_map, **kwargs)

    def _load_excluded_episode_indices(self) -> set[int]:
        """The processed manifest is the complete training population."""
        return set()

    def _resolve_cameras(self, info: dict):
        """Resolve cameras by info.features priority, record robot_type, and
        detect whether this bucket carries unified gripper columns.

        Dexterous-hand buckets have no ``gripper_open_scale_*`` columns; for
        those we shadow ``NEEDED_COLS`` (drop the grip columns so the parquet
        read doesn't fail) and set an instance ``ACTION_DIM_MASK`` that masks
        the two grip slots out of supervision. The base ``__init__`` reads
        ``self.ACTION_DIM_MASK`` after this hook (incl. the unify-mask build),
        so setting it here is the supported extension point.
        """

        features = info.get("features", {})
        head, left_wrist, right_wrist = _resolve_robocoin_cameras(features)
        if head is None:
            raise ValueError(f"No head camera found in {self._dataset_id}")
        self._robot_type = info.get("robot_type", "unknown")
        self._has_grip = all(c in features for c in _GRIP_COLS)
        if self._has_grip:
            self.NEEDED_COLS = _NEEDED_COLS
        elif self._dex_unify:
            self.NEEDED_COLS = _DEX_UNIFY_COLS
        else:
            self.NEEDED_COLS = _BASE_COLS
            self.ACTION_DIM_MASK = GRIP_EXCLUDED_DIM_MASK
        return head, left_wrist, right_wrist

    def _load_stats(self, info: dict):
        """Read the supplied family's stored EEF and optional hand statistics."""
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        path = find_dataset_stats(self._dataset_dir, f"stats_{self._robot_type}.json", family_root=self._stats_root)
        raw = _read_robot_stats(str(path))
        eef = raw.get("eef")
        stored_robot_type = raw.get("robot_type", eef.get("robot_type") if isinstance(eef, dict) else None)
        if stored_robot_type is not None and stored_robot_type != self._robot_type:
            raise ValueError(
                f"RoboCOIN stats {path} robot_type={stored_robot_type!r} does not match "
                f"bucket {self._dataset_id} robot_type={self._robot_type!r}"
            )
        eef_stats = _materialize_robot_stats(
            eef, dim=_ACTION_DIM, mode=self._normalize_mode, source=f"{path}:eef", rot6d=True
        )
        if not self._dex_unify:
            return eef_stats

        hand = raw.get("hand")
        if not isinstance(hand, dict):
            raise ValueError(f"RoboCOIN dexterous-hand bucket {self._dataset_id} requires hand stats in {path}")
        k_left, k_right = self._k_left, self._k_right
        dof = (hand.get("dof_left"), hand.get("dof_right"))
        if dof != (k_left, k_right):
            raise ValueError(
                f"RoboCOIN hand stats {path} DOF left/right={dof} do not match "
                f"bucket {self._dataset_id} finger DOF {(k_left, k_right)}"
            )
        hand_stats = _materialize_robot_stats(
            hand, dim=k_left + k_right, mode=self._normalize_mode, source=f"{path}:hand"
        )
        return {
            key: np.concatenate(
                [eef_stats[key][0:9], hand_stats[key][0:k_left], eef_stats[key][10:19], hand_stats[key][k_left:]]
            ).astype(np.float32)
            for key in STAT_KEYS
        }

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply per-bucket normalization to a (..., 20) array.

        Thin wrapper around :func:`apply_normalization` — kept on the class to
        preserve the call sites in the action/proprio hooks and the test suite.
        Returns the input untouched when ``normalize_mode`` is null / stats
        were not loaded.
        """

        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _grip_or_zeros(self, win, col: str, n: int) -> np.ndarray:
        """Stack ``n`` rows of a gripper column, or zeros for dex-hand buckets.

        Dex-hand buckets lack ``gripper_open_scale_*``; the (n, 2) zero fill
        flows through ``eef14_to_eef20`` into the two grip slots, which are
        masked out of supervision by ``GRIP_EXCLUDED_DIM_MASK``.
        """

        if self._has_grip:
            return np.stack(win[col].values[:n]).astype(np.float32)
        return np.zeros((n, 2), dtype=np.float32)

    def _dex_raw(self, eef12: np.ndarray, raw_arr: np.ndarray, fidx) -> np.ndarray:
        """Assemble + normalize a dex-hand raw action/state vector.

        ``[L_pos(3), L_rot6d(6), L_fingers(kL), R_pos(3), R_rot6d(6), R_fingers(kR)]``
        — pose from ``eef_sim_pose`` (grip dropped), fingers sliced from the raw
        ``action`` / ``observation.state`` array. Width = 18 + kL + kR; the base
        unify scatter places these into the 80-D pose + hand slots.
        """

        pose20 = eef14_to_eef20(eef12, np.zeros((len(eef12), 2), dtype=np.float32))
        l_pose, r_pose = pose20[:, 0:9], pose20[:, 10:19]
        fL, fR = fidx
        l_fing = raw_arr[:, fL].astype(np.float32)
        r_fing = raw_arr[:, fR].astype(np.float32)
        raw = np.concatenate([l_pose, l_fing, r_pose, r_fing], axis=-1)
        return self._normalize_array(raw)

    def _action_20d(self, win) -> np.ndarray:
        eef_action = np.stack(win["eef_sim_pose_action"].values).astype(np.float32)
        if self._dex_unify:
            raw_arr = np.stack(win["action"].values).astype(np.float32)
            return self._dex_raw(eef_action, raw_arr, self._fidx_act)
        grip_action = self._grip_or_zeros(win, "gripper_open_scale_action", len(eef_action))
        return self._normalize_array(eef14_to_eef20(eef_action, grip_action))

    def _proprio_20d(self, win) -> np.ndarray:
        eef_state = np.stack(win["eef_sim_pose_state"].values[:1]).astype(np.float32)
        if self._dex_unify:
            raw_arr = np.stack(win["observation.state"].values[:1]).astype(np.float32)
            return self._dex_raw(eef_state, raw_arr, self._fidx_state)
        grip_state = self._grip_or_zeros(win, "gripper_open_scale_state", len(eef_state))
        return self._normalize_array(eef14_to_eef20(eef_state, grip_state))

    @property
    def robot_type(self):
        return self._robot_type

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiRobotCOINDataset


class MultiRobotCOINDataset(MultiLeRobotV3Reader):
    """Aggregate of N RoboCOIN per-task buckets across multiple robot types."""

    def __init__(self, buckets: List[RoboCOINDataset]):
        super().__init__(buckets)
        robot_types = set(b.robot_type for b in self._buckets)
        logger.info(
            "MultiRobotCOINDataset: %d datasets, %d windows, %d robot types: %s",
            len(self._buckets),
            len(self),
            len(robot_types),
            sorted(robot_types),
        )

    @property
    def action_dim(self):
        return self._buckets[0].action_dim if self._buckets else _ACTION_DIM

    @classmethod
    def from_config(cls, config, split: str = "train"):
        return RoboCOINDataset.from_config(config, split)


__all__ = ["RoboCOINDataset", "MultiRobotCOINDataset"]
