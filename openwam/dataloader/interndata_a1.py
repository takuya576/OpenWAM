"""InternData-A1 LeRobot v3 reader with real action/state supervision.

The reader auto-detects single-arm and bimanual schemas from ``info.features``.
Pose fields are ``xyz + quaternion(wxyz)`` in the shared robot frame and refer
to the rigid arm-side controller attachment, before any task-specific TCP.
Quaternions are explicitly reordered to xyzw before conversion to rot6d.

The canonical EEF layout is ``[L_xyz(3), L_rot6d(6), L_grip,
R_xyz(3), R_rot6d(6), R_grip]``.  Single-arm Franka buckets occupy the left
half and mask the zero-padded right half.  Gripper positions are divided by
the declared full-open stroke so the emitted value is an aperture fraction in
``[0, 1]``; supported hardware variants are detected from bucket statistics.

Actions are next-state relabels and are read row-aligned.  Episode-terminal
clamped actions are excluded from supervision.  Normalization is pooled per
embodiment, with rot6d and padded dimensions pinned to identity.  Bucket
discovery is recursive because published task buckets have variable depth.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.bases.multi_lerobot_v3_reader import MultiLeRobotV3Reader
from openwam.dataloader.utils.eef import (
    ARM10_DIM,
    EEF_DIM,
    LEFT_ARM_DIM_MASK,
    assert_unit_quaternion,
    quat_wxyz_to_rot6d,
)
from openwam.dataloader.utils.lerobotv3 import build_multibucket
from openwam.dataloader.utils.normalization import apply_normalization, find_dataset_stats, materialize_eef_stats

logger = logging.getLogger(__name__)

_ACTION_DIM = EEF_DIM


# Distinguish a missing config key from one explicitly set to null.
_CONFIG_SENTINEL = object()

# Public info.json robot_type values mapped to stats-file suffixes.
ROBOT_TYPE_TO_EMBODIMENT: Dict[str, str] = {
    "Franka": "franka",
    "ARX Lift-2": "lift2",
    "Genie-1": "genie1",
    "AgileX Split Aloha": "split_aloha",
}


# Convert native gripper positions to aperture fractions (0 closed, 1 open).
GRIPPER_FULL_OPEN = {
    "franka": 0.08,
    "lift2": 0.088,
    "genie1": 5.74,
    "split_aloha": 0.1,
}

# Some embodiments publish more than one supported gripper stroke.
GRIPPER_ALT_FULL_OPEN = {
    "franka": 1.0,
}


# Values far beyond the selected stroke are surfaced but remain non-fatal.
_GRIPPER_SANE_MAX = 1.25


def resolve_gripper_scale(bucket_dir: Path, embodiment: str, grip_col: str) -> float:
    """Return the divisor mapping ``grip_col`` in this bucket onto [0, 1].

    Reads the bucket's own ``meta/stats.json`` (published with every LeRobot v3
    bucket, so this costs one small JSON read at construction). For an embodiment
    with a single gripper this is just the declared stroke. For one shipping two
    variants (franka: panda 0.08 m vs Robotiq 1.0) the observed max is matched to
    whichever declared stroke it is closer to **in log space** — the two differ by
    12.5x, so the assignment is unambiguous even for a bucket that only ever
    half-opens (0.04 is 2x from 0.08 but 25x from 1.0).

    Falls back to the primary stroke when stats are missing or the column is
    absent, which is the majority regime for every embodiment.

    Selecting the alt stroke is never silent — it rescales the whole bucket's
    gripper dim by 12.5x off a single order statistic, so it is logged, and
    logged at WARNING when the bucket's ``mean`` does not corroborate the pick.
    """

    primary = GRIPPER_FULL_OPEN.get(embodiment, 1.0)
    alt = GRIPPER_ALT_FULL_OPEN.get(embodiment)
    stats_path = bucket_dir / "meta" / "stats.json"
    try:
        with open(stats_path) as f:
            blk = json.load(f)[grip_col]
        observed_max = float(np.ravel(blk["max"])[0])

        try:
            observed_mean = float(np.ravel(blk["mean"])[0])
        except (KeyError, TypeError, IndexError, ValueError):
            observed_mean = float("nan")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        logger.debug(
            "InternData-A1: no usable %s max in %s; assuming the %s stroke %.4f",
            grip_col,
            stats_path,
            embodiment,
            primary,
        )
        return primary

    scale = primary
    if alt is not None and observed_max > 0:
        if abs(np.log(observed_max / alt)) < abs(np.log(observed_max / primary)):
            scale = alt

            if observed_mean > primary:
                logger.info(
                    "InternData-A1 %s: %s max %.4f -> alt (second-variant) stroke %.4f for %r "
                    "(mean %.4f corroborates).",
                    bucket_dir,
                    grip_col,
                    observed_max,
                    alt,
                    embodiment,
                    observed_mean,
                )
            else:
                logger.warning(
                    "InternData-A1 %s: %s max %.4f selected the alt stroke %.4f for %r, but mean "
                    "%.4f does not clear the primary stroke %.4f — if that max is a single glitch "
                    "row this bucket's gripper is being squashed by %.1fx. Verify the bucket.",
                    bucket_dir,
                    grip_col,
                    observed_max,
                    alt,
                    embodiment,
                    observed_mean,
                    primary,
                    alt / primary,
                )

    if observed_max / scale > _GRIPPER_SANE_MAX:
        logger.warning(
            "InternData-A1 %s: bucket stats %s max %.4f is %.2fx the assumed "
            "full-open stroke %.4f for %r; verify the gripper range or update GRIPPER_FULL_OPEN.",
            bucket_dir,
            grip_col,
            observed_max,
            observed_max / scale,
            scale,
            embodiment,
        )
    return scale


_BIMANUAL_COLS: Tuple[str, ...] = (
    "states.left_ee_to_robot_pose",
    "states.left_gripper.position",
    "states.right_ee_to_robot_pose",
    "states.right_gripper.position",
    "actions.left_ee_to_robot_pose",
    "actions.left_gripper.position",
    "actions.right_ee_to_robot_pose",
    "actions.right_gripper.position",
    "task_index",
)

_SINGLE_ARM_COLS: Tuple[str, ...] = (
    "states.ee_to_robot_pose",
    "states.gripper.position",
    "actions.ee_to_robot_pose",
    "actions.gripper.position",
    "task_index",
)


_BIMANUAL_SIDES = {
    "state": (
        ("states.left_ee_to_robot_pose", "states.left_gripper.position"),
        ("states.right_ee_to_robot_pose", "states.right_gripper.position"),
    ),
    "action": (
        ("actions.left_ee_to_robot_pose", "actions.left_gripper.position"),
        ("actions.right_ee_to_robot_pose", "actions.right_gripper.position"),
    ),
}
_SINGLE_ARM_SIDES = {
    "state": (("states.ee_to_robot_pose", "states.gripper.position"), None),
    "action": (("actions.ee_to_robot_pose", "actions.gripper.position"), None),
}


_HEAD_CAMERA = "images.rgb.head"

_LEFT_WRIST_BIMANUAL = "images.rgb.hand_left"
_RIGHT_WRIST_BIMANUAL = "images.rgb.hand_right"
_WRIST_SINGLE_ARM = "images.rgb.hand"


def detect_arm_layout(features: Dict[str, Any]) -> str:
    """Return ``"bimanual"`` or ``"single_arm"`` from an ``info.features`` dict.

    Detection keys off the pose feature this reader actually consumes rather
    than ``robot_type``, so a bucket with an unlisted robot_type still loads as
    long as its schema is one of the two known shapes.
    """

    if "states.left_ee_to_robot_pose" in features and "states.right_ee_to_robot_pose" in features:
        return "bimanual"
    if "states.ee_to_robot_pose" in features:
        return "single_arm"
    raise ValueError(
        "InternData-A1: info.features exposes neither the bimanual "
        "(states.left_ee_to_robot_pose + states.right_ee_to_robot_pose) nor the "
        "single-arm (states.ee_to_robot_pose) EEF schema; got keys "
        f"{sorted(k for k in features if k.startswith('states.'))}"
    )


def embodiment_key(robot_type: str, arm_layout: str) -> str:
    """Map ``info.robot_type`` to the short key used in the stats filename.

    Unknown robot types fall back to a slug of the raw string so a newly added
    embodiment gets its own stats bucket instead of silently borrowing another
    one's scale. A matching precomputed stats file must exist in the dataset
    tree before this embodiment can be normalized.
    """

    if robot_type in ROBOT_TYPE_TO_EMBODIMENT:
        return ROBOT_TYPE_TO_EMBODIMENT[robot_type]
    slug = "".join(c if c.isalnum() else "_" for c in str(robot_type).lower()).strip("_") or "unknown"
    logger.warning(
        "InternData-A1: unrecognized robot_type %r (%s layout) -> stats key %r. "
        "Provide matching precomputed normalization statistics for this embodiment.",
        robot_type,
        arm_layout,
        slug,
    )
    return slug


def discover_a1_buckets(root: Path) -> List[Path]:
    """Recursively find every LeRobot bucket (a dir holding ``meta/info.json``).

    Buckets occur at variable depth, so prune each branch when its metadata is
    found. Follow symlinks while deduplicating physical directories to avoid
    duplicate training samples and cycles. Sorted traversal keeps bucket IDs
    and subsampling reproducible. Hidden staging directories are skipped.
    """

    out: List[Path] = []
    seen: set = set()
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
        except OSError:
            dirnames[:] = []
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            dirnames[:] = []
            continue
        seen.add(key)
        if os.path.isfile(os.path.join(dirpath, "meta", "info.json")):
            out.append(Path(dirpath))
            dirnames[:] = []
            continue

        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in ("data", "videos"))
    return sorted(out)


def validate_manifest_ranges(from_idx, to_idx, lengths, episode_idx, who: str) -> None:
    """Require the cleaned manifest to describe consecutive physical rows."""
    starts = np.asarray(from_idx, dtype=np.int64)
    ends = np.asarray(to_idx, dtype=np.int64)
    sizes = np.asarray(lengths, dtype=np.int64)
    ids = np.asarray(episode_idx, dtype=np.int64)
    malformed = np.flatnonzero((starts < 0) | (sizes <= 0) | (ends - starts != sizes))
    if malformed.size:
        i = int(malformed[0])
        raise ValueError(
            f"{who}: episode {int(ids[i])} spans [{int(starts[i])}, {int(ends[i])}) but declares length={int(sizes[i])}"
        )
    if starts.size and (starts[0] != 0 or np.any(starts[1:] != ends[:-1])):
        raise ValueError(f"{who}: Overlapping manifest ranges or missing rows in the cleaned dataset")


class InternDataA1Dataset(LeRobotV3Reader):
    """Single-bucket reader for one InternData-A1 v3.0 task (LeRobot v3).

    Emits the canonical 20-D ``xyz + rot6d + gripper`` bimanual EEF with real
    action / proprio supervision. Bimanual buckets fill all 20 dims; franka
    buckets fill ``[0:10)`` and mask the rest. All window / offset / video /
    prompt machinery is inherited from :class:`LeRobotV3Reader`.
    """

    DATASET_NAME = "InternDataA1"
    ACTION_DIM = _ACTION_DIM

    NEEDED_COLS = _BIMANUAL_COLS
    PROMPT_SOURCE = "task_index"
    PROMPT_FILE_REQUIRED = True
    STATS_DIM = _ACTION_DIM
    STATS_STRICT_MINMAX = False
    DEFAULT_NORMALIZE_MODE = "quantile"
    STRICT_BUCKET_LOADING = True

    DEPLOY_ACTION_MODE = None

    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS

    def _load_excluded_episode_indices(self) -> set[int]:
        """The cleaned dataset already contains exactly the retained episodes."""
        return set()

    def _resolve_cameras(self, info: dict):
        """Detect the arm layout, then pick cameras and per-bucket columns.

        Runs before the episode index / stats / data reads, which is the
        supported point to set instance ``NEEDED_COLS`` and ``ACTION_DIM_MASK``
        (the base consults both after this hook, including for the unify mask).
        """

        features = info.get("features", {}) or {}
        self._arm_layout = detect_arm_layout(features)
        self._robot_type = info.get("robot_type", "unknown")
        self._embodiment = embodiment_key(self._robot_type, self._arm_layout)

        if self._arm_layout == "bimanual":
            self.NEEDED_COLS = _BIMANUAL_COLS
            self._sides = _BIMANUAL_SIDES

            self.ACTION_DIM_MASK = None
            left_wrist, right_wrist = _LEFT_WRIST_BIMANUAL, _RIGHT_WRIST_BIMANUAL
        else:
            self.NEEDED_COLS = _SINGLE_ARM_COLS
            self._sides = _SINGLE_ARM_SIDES

            self.ACTION_DIM_MASK = LEFT_ARM_DIM_MASK

            left_wrist, right_wrist = _WRIST_SINGLE_ARM, None

        missing = [c for c in self.NEEDED_COLS if c != "task_index" and c not in features]
        if missing:
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: {self._arm_layout} layout is missing "
                f"feature(s) {missing} in info.json."
            )

        self._grip_scale = tuple(
            resolve_gripper_scale(self._dataset_dir, self._embodiment, spec[1]) if spec is not None else 1.0
            for spec in self._sides["state"]
        )

        head = _HEAD_CAMERA if _HEAD_CAMERA in features else None
        if head is None:
            raise ValueError(
                f"InternData-A1 bucket {self._dataset_id}: no {_HEAD_CAMERA!r} camera in info.features "
                f"(has {sorted(k for k in features if k.startswith('images.'))})"
            )

        if left_wrist is not None and left_wrist not in features:
            left_wrist = None
        if right_wrist is not None and right_wrist not in features:
            right_wrist = None
        return head, left_wrist, right_wrist

    def _add_data_offsets(self, eps) -> None:
        """Use canonical v3 file IDs and contiguous manifest row ranges."""
        validate_manifest_ranges(
            eps["dataset_from_index"].to_numpy(),
            eps["dataset_to_index"].to_numpy(),
            eps["length"].to_numpy(),
            eps["episode_index"].to_numpy(),
            f"{self.DATASET_NAME}({self._dataset_id})",
        )
        super()._add_data_offsets(eps)

    def _add_episode_offsets(self, eps) -> None:
        """Read camera-frame offsets from the cleaned file-local timestamps.

        ``round(from_timestamp * fps)`` locates the same native frames as the
        metadata. Missing timestamps and inconsistent spans fail before sampling.
        """

        super()._add_episode_offsets(eps)
        for cam in self._video_cameras():
            col = f"videos/{cam}/from_timestamp"
            if col not in eps.columns:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has no "
                    f"{col!r} in meta/episodes, so its frame offset cannot be resolved."
                )
            if not np.isfinite(self._fps) or self._fps <= 0:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): fps={self._fps!r} is not a "
                    "positive finite number, so timestamps cannot be converted to frames."
                )
            ts = eps[col].to_numpy().astype(np.float64)
            if not np.isfinite(ts).all() or (ts < 0).any():
                bad = int((~np.isfinite(ts)).sum() + (ts < 0).sum())
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has {bad} "
                    "non-finite or negative from_timestamp values; casting those would "
                    "produce INT64_MIN or negative frame indices and read black slots."
                )
            frames = ts * self._fps
            if np.abs(frames - np.rint(frames)).max() > 0.25:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} has "
                    "from_timestamp values that are not on frame boundaries at "
                    f"fps={self._fps}; the offsets would be rounded onto neighbouring frames."
                )
            if frames.max() > 2**53:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} timestamp * fps "
                    "exceeds the exactly-representable integer range; the cast would wrap."
                )
            off = np.rint(frames).astype(np.int64)

            ck = f"videos/{cam}/chunk_index"
            fk = f"videos/{cam}/file_index"
            if fk in eps.columns:
                ep_len = eps["length"].to_numpy().astype(np.int64)
                chunk_of = (
                    eps[ck].to_numpy().astype(np.int64) if ck in eps.columns else np.zeros(len(off), dtype=np.int64)
                )
                file_of = eps[fk].to_numpy().astype(np.int64)
                ep_vals = eps["episode_index"].to_numpy()
                shard_key = np.stack([chunk_of, file_of], axis=1)
                for shard in np.unique(shard_key, axis=0):
                    m = np.flatnonzero((chunk_of == shard[0]) & (file_of == shard[1]))
                    order = m[np.argsort(off[m], kind="stable")]
                    for a, b in zip(order, order[1:]):
                        if off[a] + ep_len[a] > off[b]:
                            raise ValueError(
                                f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r}, video "
                                f"shard chunk-{int(shard[0]):03d}/file-{int(shard[1]):03d}: "
                                f"episode {int(ep_vals[a])} covers frames "
                                f"[{int(off[a])}, {int(off[a] + ep_len[a])}) which overlaps "
                                f"episode {int(ep_vals[b])} starting at {int(off[b])}. One of "
                                "them would read the other's frames."
                            )

            tcol = f"videos/{cam}/to_timestamp"
            if tcol in eps.columns:
                span = (eps[tcol].to_numpy().astype(np.float64) - ts) * self._fps
                declared = eps["length"].to_numpy().astype(np.int64)

                bad = np.flatnonzero(~np.isfinite(span) | (np.abs(span - declared) > 0.5))
                if bad.size:
                    i = int(bad[0])
                    raise ValueError(
                        f"{self.DATASET_NAME}({self._dataset_id}): camera {cam!r} episode "
                        f"{int(eps['episode_index'].to_numpy()[i])} spans {span[i]:.2f} video "
                        f"frames between from/to_timestamp but declares length="
                        f"{int(declared[i])}. The manifest describes two different episodes."
                    )
            eps[self._video_offset_col(cam)] = off

    def _train_min_window_len(self) -> int:
        """Require >= 2 rows so every train window keeps >= 1 supervised step.

        A 1-row episode would lose its only action step to the episode-boundary
        drop in :meth:`_n_supervised_action_steps` and yield an all-masked sample.
        """

        return 2

    def _n_supervised_action_steps(self, actual_raw_len: int) -> int:
        """Drop the clamped final action of an episode-truncated window.

        ``actions[t] == states[t+1]`` holds for every row except the episode's
        last, where the relabeling has no successor and repeats the previous
        target instead. ``actual_raw_len < num_frames`` means the window ran into
        the episode end, so its last row carries that fabricated target.
        """

        if actual_raw_len >= self._num_frames:
            return actual_raw_len
        return max(0, actual_raw_len - 1)

    def _load_stats(self, info: dict):
        """Read the existing per-embodiment EEF statistics from the dataset tree."""
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        filename = f"stats_{self._embodiment}.json"
        stats_path = find_dataset_stats(self._dataset_dir, filename, family_root=self._stats_root)
        with open(stats_path) as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or not isinstance(raw.get("eef"), dict):
            raise ValueError(f"InternData-A1: {stats_path} must contain an 'eef' statistics object")
        if raw.get("embodiment", self._embodiment) != self._embodiment:
            raise ValueError(f"InternData-A1: {stats_path} has statistics for a different embodiment")
        eef_raw = raw["eef"]
        for key in ("mean", "std", "min", "max", "q01", "q99"):
            if key not in eef_raw:
                continue
            values = np.asarray(eef_raw[key], dtype=np.float64)
            if values.shape != (_ACTION_DIM,):
                raise ValueError(
                    f"InternData-A1 bucket {self._dataset_id}: 'eef' stats {key!r} shape "
                    f"{values.shape} in {stats_path} != expected {_ACTION_DIM} dimensions"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"InternData-A1: {stats_path}: eef.{key} contains non-finite values")
        return materialize_eef_stats(
            eef_raw,
            self._normalize_mode,
            dim=_ACTION_DIM,
            strict_minmax=False,
            source_hint=f"{stats_path}: eef.*",
            force_rot6d_identity=True,
        )

    def _post_init(self, info: dict) -> None:
        """Sanity-check the quaternion magnitude on a small sample of rows.

        Catches a wrong field routed into the pose slot / un-normalized
        quaternions. It canNOT catch a wxyz<->xyzw reorder (both are unit-norm);
        that convention is fixed by the dataset's own ``names`` metadata
        (``quaternion.w`` first) and covered by the reader's unit tests.
        """

        if len(self._eps_df) == 0:
            return
        row = self._eps_df.iloc[0]
        try:
            table = self._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
            pose_col = self._sides["state"][0][0]
            sample = np.stack(table.slice(0, 64).to_pandas()[pose_col].values).astype(np.float32)
        except Exception as e:
            logger.debug("%s(%s): quaternion probe skipped (%s)", self.DATASET_NAME, self._dataset_id, e)
            return
        assert_unit_quaternion(sample[:, 3:7])

    def _arm10(self, win, spec, n: int, grip_scale: float) -> np.ndarray:
        """Build one arm's ``(n, 10)`` ``[xyz(3), rot6d(6), grip(1)]`` block.

        ``grip_scale`` maps the raw gripper column onto a normalized aperture in
        [0, 1] — see :func:`resolve_gripper_scale`.
        """

        pose_col, grip_col = spec
        pose = np.stack(win[pose_col].values[:n]).astype(np.float32)
        grip = np.stack(win[grip_col].values[:n]).astype(np.float32).reshape(n, 1) / grip_scale
        rot6d = quat_wxyz_to_rot6d(pose[:, 3:7])
        return np.concatenate([pose[:, 0:3], rot6d, grip], axis=-1)

    def _eef20(self, win, kind: str, n: int) -> np.ndarray:
        """Assemble the ``(n, 20)`` bimanual EEF for ``kind`` in {state, action}.

        Single-arm buckets fill ``[0:10)`` and leave ``[10:20)`` zero; those dims
        are excluded from supervision by ``ACTION_DIM_MASK`` and pinned to
        identity stats so normalization leaves the zeros untouched.
        """

        left_spec, right_spec = self._sides[kind]
        out = np.zeros((n, _ACTION_DIM), dtype=np.float32)
        out[:, :ARM10_DIM] = self._arm10(win, left_spec, n, self._grip_scale[0])
        if right_spec is not None:
            out[:, ARM10_DIM:] = self._arm10(win, right_spec, n, self._grip_scale[1])
        return out

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Apply this bucket's per-embodiment normalization to a ``(..., 20)`` array."""
        return apply_normalization(arr, self._normalization_stats, self._normalize_mode)

    def _action_20d(self, win) -> np.ndarray:
        n = len(win)
        return self._normalize_array(self._eef20(win, "action", n))

    def _proprio_20d(self, win) -> np.ndarray:
        return self._normalize_array(self._eef20(win, "state", 1))

    @property
    def robot_type(self) -> str:
        return self._robot_type

    @property
    def embodiment(self) -> str:
        return self._embodiment

    @property
    def arm_layout(self) -> str:
        return self._arm_layout

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiInternDataA1Dataset

    @classmethod
    def from_config(cls, config, split: str = "train") -> Any:
        """Build a cleaned bucket or all recursively discovered family buckets."""
        from openwam.dataloader.utils import get_cfg as _get

        dataset_dir = _get(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError(f"{cls.__name__}: missing dataset_dir")
        root = Path(dataset_dir)
        common: Dict[str, Any] = {"split": split}
        for key in cls.CONFIG_KEYS:
            value = _get(config, key, _CONFIG_SENTINEL)
            if value is _CONFIG_SENTINEL or (value is None and key != "normalize_mode"):
                continue
            common[key] = value

        if (root / "meta" / "info.json").is_file():
            kwargs = dict(common)
            dataset_id = _get(config, "dataset_id")
            if dataset_id is not None:
                kwargs["dataset_id"] = dataset_id
            total_hours = _get(config, "total_hours")
            if total_hours is not None:
                kwargs["max_hours"] = float(total_hours)
                kwargs["subsample_seed"] = int(_get(config, "seed", 42))
            return cls(dataset_dir=str(root), **kwargs)

        if not root.is_dir():
            raise FileNotFoundError(f"{cls.__name__}: {root} does not exist")
        sub_dirs = discover_a1_buckets(root)
        if not sub_dirs:
            raise FileNotFoundError(f"{cls.__name__}: no buckets with meta/info.json found under {root}")
        logger.info("%s.from_config: root mode, %d buckets under %s", cls.__name__, len(sub_dirs), root)

        common["stats_root"] = str(root)
        return build_multibucket(
            cls,
            sub_dirs,
            common,
            base_seed=int(_get(config, "seed", 42)),
            total_hours=_get(config, "total_hours"),
            wrapper_cls=cls._multibucket_wrapper(),
            source_name=cls.__name__,
            strict=cls.STRICT_BUCKET_LOADING,
            per_bucket_kwargs=lambda sub: {"dataset_id": sub.relative_to(root).as_posix()},
        )


class MultiInternDataA1Dataset(MultiLeRobotV3Reader):
    """Aggregate InternData-A1 task buckets across supported embodiments."""

    def __init__(self, buckets):
        super().__init__(buckets)
        counts: Dict[str, int] = {}
        windows: Dict[str, int] = {}
        for b in self._buckets:
            emb = getattr(b, "embodiment", "unknown")
            counts[emb] = counts.get(emb, 0) + 1
            windows[emb] = windows.get(emb, 0) + len(b)
        logger.info(
            "MultiInternDataA1Dataset: %d buckets, %d windows | %s",
            len(self._buckets),
            len(self),
            ", ".join(f"{k}: {counts[k]} buckets/{windows[k]} windows" for k in sorted(counts)),
        )
        self._embodiment_bucket_counts = counts

    @property
    def embodiment_bucket_counts(self) -> Dict[str, int]:
        return dict(self._embodiment_bucket_counts)


__all__ = [
    "InternDataA1Dataset",
    "MultiInternDataA1Dataset",
    "ROBOT_TYPE_TO_EMBODIMENT",
    "GRIPPER_FULL_OPEN",
    "GRIPPER_ALT_FULL_OPEN",
    "detect_arm_layout",
    "embodiment_key",
    "discover_a1_buckets",
    "resolve_gripper_scale",
]
