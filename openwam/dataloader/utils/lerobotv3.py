"""Shared metadata, split and footage-budget helpers for LeRobot v3 readers.

Readers consume canonical episode manifests and compute file-local offsets
before applying train/validation splits or whole-episode footage budgets.
Final pretraining exports need no additional frame-selection artifacts.
``build_multibucket`` preserves strict pretraining failures while allowing
benchmark readers to retain their existing per-bucket error policy.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


_DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


class DataContractError(RuntimeError):
    """A per-bucket failure that proves the DATA is broken, not the environment.

    :func:`build_multibucket` deliberately tolerates a bucket that fails to
    construct — a truncated shard, an NFS blip, a stale stats file — by logging
    a warning and dropping it, so one bad corner of a large multi-bucket root cannot
    abort a training launch. That tolerance is wrong for a failure that means
    the dataset itself violates the reader's contract (a prompt table that has
    diverged from the episodes table, a mis-typed text column): dropping such a
    bucket silently removes a slice of the training set behind a single WARNING,
    which is exactly the silent data loss the warning was meant to surface.

    A reader raises this (instead of a plain exception) when it has PROVEN a
    data-contract violation. ``_build_one`` re-raises it so the launch fails
    loudly with the reader's own diagnostic, in root mode as well as in
    single-bucket mode. Environmental failures must keep using ordinary
    exceptions so they stay tolerated.
    """


def parse_info_json(dataset_dir: Path) -> dict:
    """Read ``<dataset_dir>/meta/info.json`` and fill in default templates.

    Returns the raw dict augmented with default ``data_path`` /
    ``video_path`` entries when the file omits them.
    """
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info.setdefault("data_path", _DEFAULT_DATA_PATH)
    info.setdefault("video_path", _DEFAULT_VIDEO_PATH)
    return info


def load_episodes_parquet(dataset_dir: Path) -> pd.DataFrame:
    """Concatenate per-shard episodes parquet into one sorted DataFrame.

    Drops any ``stats/*`` column at read time (they aren't consumed by the
    readers and skipping them halves the read cost on shards that ship
    pre-computed stats columns).

    Raises FileNotFoundError when no shard exists.
    """
    eps_paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not eps_paths:
        raise FileNotFoundError(f"No episode parquet under {dataset_dir}/meta/episodes")
    schema_cols = pq.read_schema(eps_paths[0]).names
    keep_cols = [c for c in schema_cols if not c.startswith("stats/")]
    eps_tables = [pq.read_table(p, columns=keep_cols) for p in eps_paths]
    eps = pa.concat_tables(eps_tables).to_pandas()
    return eps.sort_values("episode_index").reset_index(drop=True)


def compute_file_local_offsets(eps: pd.DataFrame, chunk_col: str, file_col: str) -> np.ndarray:
    """Return per-episode file-local cumulative offsets.

    For a contiguous-on-disk format like LeRobot v3, episode N within a
    given (chunk, file) shard starts at offset = Σ length of episodes
    earlier than N in the same shard. The lexsort key
    ``(dataset_from_index, file_col, chunk_col)`` puts episodes in shard
    order; the ``groupby([chunk_col, file_col]).cumsum() - length`` trick
    converts inclusive cumulative length to exclusive offset; the inverse
    permutation puts results back into eps's original row order.
    """
    order = np.lexsort(
        (
            eps["dataset_from_index"].to_numpy(),
            eps[file_col].to_numpy(),
            eps[chunk_col].to_numpy(),
        )
    )
    sorted_eps = eps.iloc[order]
    cum_inclusive = sorted_eps.groupby([chunk_col, file_col], sort=False)["length"].cumsum()
    cum_exclusive = (cum_inclusive - sorted_eps["length"]).to_numpy().astype(np.int64)
    out = np.empty(len(eps), dtype=np.int64)
    out[order] = cum_exclusive
    return out


def apply_info_splits(
    eps_df: pd.DataFrame,
    split: str,
    info_splits: dict,
    *,
    source_name: str = "dataset",
) -> pd.DataFrame:
    """Honor ``info.json[splits][split]`` when declared; default otherwise.

    - If ``split`` appears in ``info_splits``: expect ``"start:end"`` (int
      bounds, half-open), filter ``eps_df`` to ``episode_index ∈ [start, end)``.
    - Otherwise: ``train`` returns the full eps_df, anything else returns
      empty. Pretraining-only projects depend on the empty-val path; if
      future consumers need a different fallback they can add it here.

    Raises ValueError on a malformed split spec.
    """
    if split in info_splits:
        spec = info_splits[split]
        try:
            start_str, end_str = spec.split(":")
            start, end = int(start_str), int(end_str)
        except (ValueError, AttributeError):
            raise ValueError(f"info.json splits.{split} must be 'start:end', got {spec!r}")
        sel = eps_df[(eps_df["episode_index"] >= start) & (eps_df["episode_index"] < end)].reset_index(drop=True)
        logger.info(
            "%s: split=%s from info.json (episode_index %d:%d, %d eps)",
            source_name,
            split,
            start,
            end,
            len(sel),
        )
        return sel
    return eps_df.reset_index(drop=True) if split == "train" else eps_df.iloc[0:0].reset_index(drop=True)


def water_fill_hours(bucket_hours: List[float], total_budget: float) -> List[float]:
    """Allocate ``total_budget`` hours across N buckets via water-filling.

    Each bucket gets at most its own ``bucket_hours[i]``. Surplus from
    under-budgetable buckets (too small to absorb fair share) is
    redistributed equally among the rest. The result is parallel to
    ``bucket_hours``; the sum is min(total_budget, sum(bucket_hours)).

    Args:
        bucket_hours: List of total available hours per bucket.
        total_budget: Target total hours across all buckets. A non-positive
            value returns all-zeros; a value >= sum(bucket_hours) returns
            ``bucket_hours`` unchanged (everyone gets full).

    Returns:
        List of allocations parallel to ``bucket_hours``. Each element is
        in [0, bucket_hours[i]].

    Examples:
        >>> water_fill_hours([10.0, 10.0, 10.0], 15.0)
        [5.0, 5.0, 5.0]
        >>> water_fill_hours([0.5, 10.0, 10.0], 5.0)
        [0.5, 2.25, 2.25]
        >>> water_fill_hours([1.0, 1.0, 1.0], 100.0)   # over-budget
        [1.0, 1.0, 1.0]
    """
    n = len(bucket_hours)
    if n == 0:
        return []
    if total_budget <= 0:
        return [0.0] * n
    if total_budget >= sum(bucket_hours):
        return [float(h) for h in bucket_hours]

    alloc = [0.0] * n
    active = list(range(n))
    remaining = float(total_budget)

    while active and remaining > 1e-9:
        fair_share = remaining / len(active)
        # Buckets too small to absorb fair_share — cap them at their max.
        small = [i for i in active if bucket_hours[i] <= fair_share]
        if not small:
            # Every remaining bucket can absorb fair_share; done.
            for i in active:
                alloc[i] = fair_share
            break
        for i in small:
            alloc[i] = float(bucket_hours[i])
            remaining -= bucket_hours[i]
            active.remove(i)

    return alloc


def effective_episode_frames(eps_df: pd.DataFrame, length_col: str = "length") -> np.ndarray:
    """Return validated physical frame counts for each episode row."""
    lengths = eps_df[length_col].to_numpy(dtype=np.int64)
    if np.any(lengths < 0):
        raise ValueError(f"episode {length_col!r} values must be non-negative")
    return lengths


def subsample_episodes_by_hours(
    eps_df: pd.DataFrame,
    target_hours: float,
    fps: float,
    seed: int,
    length_col: str = "length",
    episode_frames: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Random subset totalling >= ``target_hours`` of effective footage.

    Selection: seeded permutation over rows, then greedy prefix until
    cumulative episode length >= target_hours * fps * 3600. Returns
    rows in their *original* eps_df order — important because downstream offset
    columns (``_data_row_offset`` / ``_video_frame_offset``) are computed per-row
    and must stay aligned.

    If the full eps_df already fits in target_hours, returns it unchanged
    (identity short-circuit, no permutation overhead).

    Args:
        eps_df: Episodes DataFrame with at minimum a ``length_col`` column.
        target_hours: Target total duration in hours. Must be > 0.
        fps: Frame rate (from info.json).
        seed: Random seed for the permutation.
        length_col: Name of the physical episode-length column.
        episode_frames: Optional explicit non-negative frame count per row.
            Reader-level callers use this to zero episodes that cannot produce a
            window. When omitted, :func:`effective_episode_frames` is used.

    Returns:
        Filtered DataFrame (same columns, subset of rows, in original order,
        with index reset).

    Raises:
        ValueError if target_hours <= 0.
    """
    if not np.isfinite(target_hours) or target_hours <= 0:
        raise ValueError(f"target_hours must be > 0, got {target_hours}")
    target_float = float(target_hours) * 3600.0 * float(fps)
    # ``250 / fps / 3600`` can round-trip as 250.00000000000003. Nudge one
    # representable float toward -inf before ceil so an exactly frame-aligned
    # request does not spuriously demand a sixth 50-frame episode.
    target_frames = max(1, int(np.ceil(np.nextafter(target_float, -np.inf))))
    if episode_frames is None:
        lengths = effective_episode_frames(eps_df, length_col=length_col)
    else:
        raw = np.asarray(episode_frames)
        if raw.shape != (len(eps_df),):
            raise ValueError(f"episode_frames shape {raw.shape} != ({len(eps_df)},)")
        lengths = raw.astype(np.int64)
        if np.any(raw != lengths) or np.any(lengths < 0):
            raise ValueError("episode_frames must contain non-negative integer frame counts")
    available_frames = int(lengths.sum())
    if available_frames <= 0:
        raise ValueError("episode population contains no positive sampleable frame span")
    if available_frames <= target_frames:
        return eps_df.reset_index(drop=True)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(eps_df))
    cum = 0
    selected = []
    for idx in perm:
        if lengths[idx] <= 0:
            continue
        cum += int(lengths[idx])
        selected.append(int(idx))
        if cum >= target_frames:
            break
    return eps_df.iloc[sorted(selected)].reset_index(drop=True)


def build_multibucket(
    reader_cls: type,
    sub_dirs: List[Path],
    common: Dict[str, Any],
    *,
    base_seed: int,
    total_hours: Optional[float],
    wrapper_cls: type,
    source_name: str = "dataset",
    per_bucket_kwargs: Optional[Callable[[Path], Dict[str, Any]]] = None,
    strict: bool = False,
) -> Any:
    """Construct N per-bucket readers from ``sub_dirs`` and wrap them.

    Honors an optional ``total_hours`` water-fill budget over the readers'
    available post-split population. Leaves are initialized uncapped once,
    their sampleable capacities are water-filled, and the resulting
    whole-episode subsets are applied in place. Empty buckets are filtered.
    With ``strict=True``, construction failures abort the launch; otherwise
    ordinary failures are logged and skipped. DataContractError always aborts.

    Args:
        reader_cls: the per-bucket reader class to instantiate.
        sub_dirs: discovered bucket directories (each with ``meta/info.json``).
        common: kwargs shared by every bucket (split + window/video knobs).
        base_seed: seed base for per-bucket subsample (``base_seed + i*7919``).
        total_hours: optional global hour budget; None → load full dataset.
        wrapper_cls: the ``MultiLeRobotV3Reader`` subclass to wrap buckets in.
        source_name: label for log lines.
        per_bucket_kwargs: optional callable ``(sub_dir) -> dict`` for extra
            per-bucket kwargs (unused by current readers; kept for extension).
        strict: propagate construction failures instead of silently dropping a
            bucket. Final pretraining data uses this to preserve full coverage.
    """
    if total_hours is not None:
        total_hours = float(total_hours)
        if not np.isfinite(total_hours) or total_hours <= 0:
            raise ValueError(f"{source_name} total_hours must be > 0 or null/unset; got {total_hours}.")
    else:
        logger.info("%s: total_hours unset, loading full dataset", source_name)

    def _build_one(args):
        _, sub = args
        kwargs = dict(common)
        kwargs["dataset_dir"] = str(sub)
        if per_bucket_kwargs is not None:
            kwargs.update(per_bucket_kwargs(sub))
        try:
            return reader_cls(**kwargs)
        except DataContractError:
            # Proven broken data (see DataContractError): dropping the bucket
            # would hide a slice of the training set behind a warning, so let
            # the reader's own diagnostic abort the launch.
            raise
        except Exception as e:
            if strict:
                raise
            logger.warning("%s: skipping %s: %s", source_name, sub.name, e)
            return None

    build_args = list(enumerate(sub_dirs))
    with ThreadPoolExecutor(max_workers=min(len(sub_dirs), 16)) as pool:
        results = list(pool.map(_build_one, build_args))
    loaded = [(i, sub, r) for i, (sub, r) in enumerate(zip(sub_dirs, results)) if r is not None and len(r) > 0]
    if not loaded:
        raise RuntimeError(f"All {source_name} buckets failed to load")

    if total_hours is not None:
        bucket_hours = [float(reader.effective_hours) for _, _, reader in loaded]
        total_avail = sum(bucket_hours)
        if total_hours > total_avail - 1e-3:
            # Over budget: every bucket already represents the complete effective
            # population, so applying a near-equal cap only risks a float-rounding
            # loss. Keep each initialized leaf unchanged.
            logger.warning(
                "%s total_hours=%.3f exceeds available effective footage %.3fh; "
                "loading the full post-split population.",
                source_name,
                total_hours,
                total_avail,
            )
        else:
            allocations = water_fill_hours(bucket_hours, total_hours)
            for (original_i, _, reader), allocation in zip(loaded, allocations):
                reader._apply_effective_hour_budget(allocation, base_seed + original_i * 7919)
            realized = sum(float(reader.effective_hours) for _, _, reader in loaded if len(reader) > 0)
            logger.info(
                "%s total_hours=%.3f → allocated %.3f effective h across %d buckets; whole-episode realization %.3fh.",
                source_name,
                total_hours,
                sum(allocations),
                len(loaded),
                realized,
            )

    buckets = [reader for _, _, reader in loaded if len(reader) > 0]
    if not buckets:
        raise RuntimeError(f"All {source_name} buckets became empty after effective-hour subsampling")
    # Report empty buckets and failures tolerated by non-strict benchmark readers.
    dropped = [sub.name for sub, r in zip(sub_dirs, results) if r is None or len(r) == 0]
    if dropped:
        logger.warning(
            "%s: dropped %d / %d bucket(s) during load (construction failed or empty): %s",
            source_name,
            len(dropped),
            len(sub_dirs),
            ", ".join(sorted(dropped)),
        )
    logger.info("%s: loaded %d / %d buckets", source_name, len(buckets), len(sub_dirs))
    return wrapper_cls(buckets)


__all__ = [
    "DataContractError",
    "parse_info_json",
    "load_episodes_parquet",
    "compute_file_local_offsets",
    "apply_info_splits",
    "water_fill_hours",
    "effective_episode_frames",
    "subsample_episodes_by_hours",
    "build_multibucket",
]
