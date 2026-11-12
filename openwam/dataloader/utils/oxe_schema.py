"""Canonical Euler/wrist-pose converters for DROID and VLABench."""

from __future__ import annotations

import numpy as np

from openwam.dataloader.utils.eef import euler_xyz_to_rot6d

ARM10_DIM = 10


def euler7_action_to_arm10(action: np.ndarray) -> np.ndarray:
    """``(..., 7)`` ``[x,y,z,roll,pitch,yaw,gripper]`` → ``(..., 10)`` EEF.

    Used by VLABench pose streams. Current DROID has the same
    pose layout but opposite raw gripper direction, so it must use
    :func:`droid_euler7_to_arm10` instead.
    """
    action = np.asarray(action)
    if action.ndim == 0 or action.shape[-1] != 7:
        raise ValueError(f"Euler pose/gripper input must have width 7, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("Euler pose/gripper input must contain only finite values")
    pos = action[..., 0:3]
    euler = action[..., 3:6]
    grip = action[..., 6:7]
    rot6d = euler_xyz_to_rot6d(euler.reshape(-1, 3)).reshape(action.shape[:-1] + (6,))
    return np.concatenate([pos, rot6d, grip], axis=-1).astype(np.float32)


def droid_euler7_to_arm10(value: np.ndarray) -> np.ndarray:
    """Current DROID commanded wrist ``(..., 7)`` -> canonical 10-D EEF.

    ``other_information.action_wrist_pose`` contains
    ``[x,y,z,roll,pitch,yaw,closedness]``.  The shared action space uses an
    openness scale, so only the gripper scalar is inverted; xyz and
    Euler->rot6d are identical to :func:`euler7_action_to_arm10`.
    """
    out = euler7_action_to_arm10(value)
    out[..., 9] = 1.0 - out[..., 9]
    return out


def droid_pose6_closedness_to_arm10(pose: np.ndarray, closedness: np.ndarray) -> np.ndarray:
    """DROID achieved wrist/gripper-mount pose + closedness -> 10-D EEF.

    Args:
        pose: ``(..., 6)`` from
            ``other_information.observation_gripper_pose6d`` in Euler XYZ.
            Despite the source's ``gripper`` spelling, this is the rigid
            arm-side mount pose, not the moving-finger/task TCP pose.
        closedness: ``(..., 1)`` copied from ``state[..., 6:7]``; DROID raw
            convention ``0=open, 1=closed``.

    The explicit two-input helper prevents the old TCP xyz stored in
    ``state[..., :6]`` from silently re-entering the unified arm-side frame.
    """
    pose = np.asarray(pose)
    closedness = np.asarray(closedness)
    if pose.ndim == 0 or pose.shape[-1] != 6:
        raise ValueError(f"DROID gripper-mount pose must have width 6, got shape {pose.shape}")
    if closedness.ndim == 0 or closedness.shape[-1] != 1:
        raise ValueError(f"DROID closedness must have width 1, got shape {closedness.shape}")
    if pose.shape[:-1] != closedness.shape[:-1]:
        raise ValueError(
            f"DROID pose and closedness leading shapes must match, got {pose.shape} and {closedness.shape}"
        )
    return droid_euler7_to_arm10(np.concatenate([pose, closedness], axis=-1))


__all__ = ["ARM10_DIM", "euler7_action_to_arm10", "droid_euler7_to_arm10", "droid_pose6_closedness_to_arm10"]
