"""Normalization statistics for bimanual YAM joint-space LeRobot v2.1 datasets.

Scans the ``action`` / ``observation.state`` columns of every train-split
episode (the yaml ``splits`` override, else ``info.json``'s) and writes
separate action and state blocks with min / max / mean / std / quantiles per
joint dimension, plus two YAM rules that the reader's min-max normalization
relies on:

* grippers are pinned to their physical [0, 1] range, so after min-max
  -1 = closed and +1 = open regardless of what a task happened to record;
* arm joints whose recorded span is below ``MIN_JOINT_SPAN_RAD`` are widened
  to that span around their midpoint, so an arm held still normalizes near 0
  instead of stretching sensor noise across [-1, 1].

With ``delta_action_dims`` the action block is fitted on what the reader then
serves: every chunk target ``action[t + k]`` with those dims taken relative to
``state[t]``, for ``k < delta_horizon`` within the episode. The state block
stays absolute.

Default output, named by the train range so a new split never reuses stale
stats (auto-built by the reader on first use — rank 0 computes, other ranks
wait):

    <dataset_dir>/meta/yam_normalization_stats_train_<start>-<end>.npy

CLI:
    python -m openwam.dataloader.utils.stats_computation.yam_stats_computation \\
        --dataset-dir /path/to/yam_dataset --train-split 0:55
    # delta actions over 32-step chunks, every arm joint relative:
    python -m openwam.dataloader.utils.stats_computation.yam_stats_computation \\
        --dataset-dir /path/to/yam_dataset --train-split 0:56 --delta-action-dims 0-5 7-12 --delta-horizon 32
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.yam import (
    ACTION_STATS_KEY,
    ARM_JOINT_DIMS,
    GRIPPER_CONVENTION,
    GRIPPER_DIMS,
    GRIPPER_RANGE,
    JOINT_DIM,
    MIN_JOINT_SPAN_RAD,
    STATE_STATS_KEY,
    default_stats_path,
    resolve_delta_dims,
    resolve_splits,
    train_data_fingerprint,
    train_episode_files,
)

ACTION_COLUMN = "action"
STATE_COLUMN = "observation.state"


def _feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"statistics require a non-empty 2-D array, got {values.shape}")
    quantiles = np.quantile(values, [0.01, 0.99], axis=0)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "q01": quantiles[0].tolist(),
        "q99": quantiles[1].tolist(),
    }


def _apply_yam_ranges(stats: dict) -> None:
    """In-place: pin gripper min/max to [0, 1] and floor arm-joint spans."""
    for dim in GRIPPER_DIMS:
        stats["min"][dim], stats["max"][dim] = GRIPPER_RANGE
    for dim in ARM_JOINT_DIMS:
        lo, hi = stats["min"][dim], stats["max"][dim]
        if hi - lo < MIN_JOINT_SPAN_RAD:
            mid = 0.5 * (lo + hi)
            stats["min"][dim] = mid - 0.5 * MIN_JOINT_SPAN_RAD
            stats["max"][dim] = mid + 0.5 * MIN_JOINT_SPAN_RAD


def _load_episodes(dataset_dir: Path, splits: Optional[Mapping[str, str]]) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Per-episode action and state arrays of the train split."""
    files = train_episode_files(dataset_dir, splits)
    if not files:
        raise ValueError(f"no train-split episodes in {dataset_dir}")
    actions, states = [], []
    for path in files:
        table = pq.read_table(path, columns=[ACTION_COLUMN, STATE_COLUMN])
        actions.append(np.stack(table.column(ACTION_COLUMN).to_pylist()).astype(np.float32))
        states.append(np.stack(table.column(STATE_COLUMN).to_pylist()).astype(np.float32))
    for name, episodes in ((ACTION_COLUMN, actions), (STATE_COLUMN, states)):
        for values in episodes:
            if values.shape[1] != JOINT_DIM:
                raise ValueError(f"YAM {name} must be {JOINT_DIM}-D, got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"YAM {name} contains NaN or infinity")
    return actions, states


def delta_chunk_actions(action: np.ndarray, state: np.ndarray, dims: Sequence[int], horizon: int) -> np.ndarray:
    """Every chunk target of one episode: ``action[t + k]`` with ``dims`` relative to ``state[t]``, ``k < horizon``."""
    dims = list(dims)
    rows = []
    for k in range(min(horizon, len(action))):
        target = action[k:].copy()
        target[:, dims] -= state[: len(action) - k, dims]
        rows.append(target)
    return np.concatenate(rows, axis=0)


def compute_yam_stats(
    dataset_dir: str | Path,
    splits: Optional[Mapping[str, str]] = None,
    *,
    delta_action_dims=None,
    delta_horizon: Optional[int] = None,
) -> dict:
    """Train-split scan of the YAM joint action/state columns."""
    dims = resolve_delta_dims(delta_action_dims)
    if bool(dims) != (delta_horizon is not None):
        raise ValueError("delta_action_dims and delta_horizon must be given together")
    actions, states = _load_episodes(Path(dataset_dir), splits)
    if dims:
        action_all = np.concatenate(
            [delta_chunk_actions(a, s, dims, int(delta_horizon)) for a, s in zip(actions, states)]
        )
        action_pool = "action_chunks_relative_to_state"
    else:
        action_all, action_pool = np.concatenate(actions), "action_only"
    state_all = np.concatenate(states)
    train_split = resolve_splits(dataset_dir, splits).get("train", "all")
    fingerprint = train_data_fingerprint(dataset_dir, splits)
    payload = {}
    for key, values, pool in (
        (ACTION_STATS_KEY, action_all, action_pool),
        (STATE_STATS_KEY, state_all, "state_only"),
    ):
        stats = _feature_stats(values)
        _apply_yam_ranges(stats)
        stats.update(
            {
                "num_timesteps": int(values.shape[0]),
                "train_split": train_split,
                "source_sha256": fingerprint,
                "pool": pool,
                "gripper_convention": GRIPPER_CONVENTION,
                "min_joint_span_rad": MIN_JOINT_SPAN_RAD,
            }
        )
        payload[key] = stats
    payload[ACTION_STATS_KEY]["delta_action_dims"] = list(dims)
    payload[ACTION_STATS_KEY]["delta_horizon"] = int(delta_horizon) if dims else None
    return payload


def build_and_save_yam_stats(
    dataset_dir: str | Path,
    output: str | Path | None = None,
    splits: Optional[Mapping[str, str]] = None,
    *,
    delta_action_dims=None,
    delta_horizon: Optional[int] = None,
) -> Path:
    """Compute and atomically write the stats payload; returns the output path."""
    dataset_dir = Path(dataset_dir)
    out_path = Path(output) if output else default_stats_path(dataset_dir, splits, delta_horizon=delta_horizon)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_yam_stats(dataset_dir, splits, delta_action_dims=delta_action_dims, delta_horizon=delta_horizon)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp_name, out_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="YAM LeRobot v2.1 dataset root")
    parser.add_argument(
        "--train-split",
        default=None,
        help="Half-open episode range 'start:end' (default: info.json's train split)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=".npy path (default <dataset_dir>/meta/yam_normalization_stats_train_<start>-<end>.npy)",
    )
    parser.add_argument(
        "--delta-action-dims",
        nargs="+",
        default=None,
        help="Joint dims taken relative to the chunk-start state, e.g. 0-5 7-12 (default: absolute actions)",
    )
    parser.add_argument("--delta-horizon", type=int, default=None, help="Action steps per chunk (num_frames - 1)")
    args = parser.parse_args()
    splits = {"train": args.train_split} if args.train_split else None
    out_path = build_and_save_yam_stats(
        args.dataset_dir,
        args.output,
        splits,
        delta_action_dims=args.delta_action_dims,
        delta_horizon=args.delta_horizon,
    )
    payload = np.load(out_path, allow_pickle=True).item()
    print(f"wrote {out_path}")
    for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
        block = payload[key]
        print(f"  {key}: num_timesteps={block['num_timesteps']}")
        print(f"    min={np.round(block['min'], 4).tolist()}")
        print(f"    max={np.round(block['max'], 4).tolist()}")


if __name__ == "__main__":
    main()
