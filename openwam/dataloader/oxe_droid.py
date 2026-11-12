"""Reader for the physically cleaned DROID LeRobot v3 export.

Prompts are resolved through the export's ``meta/tasks.parquet``. The rigid
arm-side wrist/gripper-mount poses retain the same base frame and Euler XYZ
conversion as the source loader; raw gripper closedness is inverted to the
shared ``0=closed, 1=open`` convention. Statistics are read from the dataset's
``meta/eef_stats.json`` and are never computed during loading.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.eef import EEF_POSE_FRAME_CONTRACT, LEFT_ARM_DIM_MASK, single_arm_20d
from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.utils.normalization import materialize_eef_stats
from openwam.dataloader.utils.oxe_schema import droid_euler7_to_arm10, droid_pose6_closedness_to_arm10

DROID_EEF_STATS_CONTRACT_KEY = "droid_eef_stats_contract"
DROID_EEF_STATS_CONTRACT_VERSION = 2
DROID_EEF_STATS_CONTRACT = {
    "schema_version": DROID_EEF_STATS_CONTRACT_VERSION,
    "pose_frame_semantics": EEF_POSE_FRAME_CONTRACT,
    "action_pose_source": "other_information.action_wrist_pose",
    "state_pose_source": "other_information.observation_gripper_pose6d",
    "state_gripper_source": "state[6]",
    "raw_gripper_semantics": "closedness:0=open,1=closed",
    "output_gripper_semantics": "openness:0=closed,1=open",
    "gripper_transform": "1-raw",
}


class OxeDroidDataset(LeRobotV3Reader):
    """Load DROID's exported rows directly, without further frame selection."""

    STRICT_BUCKET_LOADING = True
    DATASET_NAME = "DROID"
    HEAD_CAMERA = "observation.images.primary"
    LEFT_WRIST_CAMERA = "observation.images.wrist"
    RIGHT_WRIST_CAMERA = None
    NEEDED_COLS = (
        "state",
        "other_information.observation_gripper_pose6d",
        "other_information.action_wrist_pose",
        "task_index",
    )
    ACTION_DIM_MASK = LEFT_ARM_DIM_MASK
    PROMPT_FILE_REQUIRED = True
    DEFAULT_NORMALIZE_MODE = "quantile"
    STATS_FILENAME = "eef_stats.json"
    STATS_DIM = 10
    STATS_STRICT_MINMAX = True

    def _load_excluded_episode_indices(self) -> set[int]:
        """The export has already physically removed excluded episodes."""
        return set()

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        stats_path = self._dataset_dir / "meta" / self.STATS_FILENAME
        if not stats_path.is_file():
            raise FileNotFoundError(f"DROID requires its own precomputed statistics at {stats_path}.")
        with stats_path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, dict):
            raise DataContractError(f"{stats_path} must contain a statistics mapping.")
        if raw.get(DROID_EEF_STATS_CONTRACT_KEY) != DROID_EEF_STATS_CONTRACT:
            raise DataContractError(
                f"{stats_path} does not match the rigid terminal-arm pose frame and canonical gripper convention. "
                "Use the stats shipped with the cleaned DROID export."
            )
        stats = materialize_eef_stats(
            raw,
            self._normalize_mode,
            dim=self.STATS_DIM,
            strict_minmax=self.STATS_STRICT_MINMAX,
            source_hint=str(stats_path),
            force_rot6d_identity=True,
        )
        return stats

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        action = np.stack(win["other_information.action_wrist_pose"].values).astype(np.float32)
        if action.ndim != 2 or action.shape[1] != 7:
            raise DataContractError(f"DROID commanded wrist pose must have width 7, got {action.shape}.")
        return single_arm_20d(droid_euler7_to_arm10(action), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> Optional[np.ndarray]:
        pose = np.stack(win["other_information.observation_gripper_pose6d"].values[:1]).astype(np.float32)
        state = np.stack(win["state"].values[:1]).astype(np.float32)
        if state.ndim != 2 or state.shape[1] < 7:
            raise DataContractError(f"DROID state must provide gripper closedness at index 6, got {state.shape}.")
        arm10 = droid_pose6_closedness_to_arm10(pose, state[:, 6:7])
        return single_arm_20d(arm10, self._normalization_stats, self._normalize_mode)


__all__ = [
    "DROID_EEF_STATS_CONTRACT",
    "DROID_EEF_STATS_CONTRACT_KEY",
    "DROID_EEF_STATS_CONTRACT_VERSION",
    "OxeDroidDataset",
]
