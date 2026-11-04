"""Open-loop evaluation of a trained checkpoint on recorded episodes.

Every chunk is predicted from a recorded observation: the camera frames, robot
state and task prompt at chunk start ``t`` go through the policy, and its
output is scored against what was recorded next.

* Action ``k`` of the chunk predicted at ``t`` is compared with recorded action
  ``t + k``, the alignment the readers train on. Nothing is shifted to fit.
* Generated video frame ``j`` is compared with the recorded observation at
  ``t + j * video_stride``, rendered on the canvas the model sees. Frame 0 is
  the VAE reconstruction of the conditioning observation, not a prediction.

Chunks start every ``horizon`` steps and the first ``horizon`` actions of each
form the executed trajectory, as the deploy executor does with
``inference_horizon``. Every chunk restarts from a recorded observation, so this
measures prediction quality on held-out data, not closed-loop drift. As a
reference point, each chunk is also scored against holding the state it starts
from.

Reads LeRobot v2.1 recordings: one parquet and one mp4 per camera per episode.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from openwam.dataloader.utils.video_io import decode_video_frames

logger = logging.getLogger(__name__)

# Reported for identical images, so summary.json stays standard JSON.
MAX_PSNR_DB = 100.0


class ChunkPolicy(Protocol):
    """What the evaluator needs from a policy; see :class:`openwam.eval.policy.EnginePolicy`."""

    cameras: Tuple[Optional[str], ...]  # dataset camera keys in canvas order; None = empty slot
    chunk_len: int  # actions per generated chunk
    video_stride: int  # recorded steps between consecutive generated video frames

    def predict(
        self, images: Mapping[str, Image.Image], state: np.ndarray, prompt: str
    ) -> Tuple[np.ndarray, Optional[List[Image.Image]]]:
        """Return ``(actions of shape (chunk_len, D), generated video frames or None)``."""

    def canvas(self, images: Mapping[str, Image.Image], state: np.ndarray) -> Image.Image:
        """The model-input canvas for recorded frames: the reference for generated frames."""


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class Episode:
    """One recorded episode, with frames decoded only at the steps evaluation needs."""

    index: int
    prompt: str
    actions: np.ndarray  # (T, D) recorded commands
    states: np.ndarray  # (T, D) recorded robot state
    names: List[str]
    fps: float
    video_paths: Dict[str, Path]
    frame_sizes: Dict[str, Tuple[int, int]]  # camera -> (height, width)
    frames: Dict[str, Dict[int, Image.Image]] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.actions)

    def load_frames(self, steps: Sequence[int]) -> None:
        """Decode every camera at ``steps``, at full resolution as a client would send them."""
        wanted = sorted({int(s) for s in steps if 0 <= s < len(self)})
        for camera, path in self.video_paths.items():
            height, width = self.frame_sizes[camera]
            self.frames[camera] = dict(zip(wanted, decode_video_frames(str(path), wanted, height, width)))

    def images(self, step: int) -> Dict[str, Image.Image]:
        return {camera: frames[step] for camera, frames in self.frames.items()}


def read_episode(dataset_dir: str | Path, index: int, cameras: Sequence[Optional[str]]) -> Episode:
    """Read one LeRobot v2.1 episode's actions, states and prompt; frames load on demand."""
    root = Path(dataset_dir)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"open-loop evaluation reads LeRobot v2.1 datasets, got {info.get('codebase_version')!r}")
    chunk = index // int(info.get("chunks_size", 1000))
    data_path = root / info["data_path"].format(episode_chunk=chunk, episode_index=index)
    table = pq.read_table(data_path, columns=["action", "observation.state", "task_index"])
    actions = np.stack(table.column("action").to_pylist()).astype(np.float64)
    states = np.stack(table.column("observation.state").to_pylist()).astype(np.float64)
    tasks = {int(row["task_index"]): str(row["task"]) for row in _read_jsonl(root / "meta" / "tasks.jsonl")}
    features = info["features"]
    names = features["action"].get("names") or [f"action_{i}" for i in range(actions.shape[1])]
    video_paths, frame_sizes = {}, {}
    for camera in cameras:
        if camera is None:
            continue
        if features.get(camera, {}).get("dtype") != "video":
            raise ValueError(f"camera {camera!r} is not a video feature of {root}")
        video_paths[camera] = root / info["video_path"].format(
            episode_chunk=chunk, video_key=camera, episode_index=index
        )
        height, width = features[camera]["shape"][:2]
        frame_sizes[camera] = (int(height), int(width))
    return Episode(
        index=index,
        prompt=tasks[int(table.column("task_index")[0].as_py())],
        actions=actions,
        states=states,
        names=list(names),
        fps=float(info["fps"]),
        video_paths=video_paths,
        frame_sizes=frame_sizes,
    )


def chunk_starts(n_steps: int, horizon: int) -> List[int]:
    return list(range(0, n_steps, horizon))


def frame_steps(n_steps: int, horizon: int, chunk_len: int, video_stride: Optional[int]) -> List[int]:
    """Steps whose frames are needed: chunk starts, plus each chunk's video span when scoring video."""
    steps = set(chunk_starts(n_steps, horizon))
    if video_stride:
        for start in chunk_starts(n_steps, horizon):
            steps.update(range(start, min(start + chunk_len, n_steps - 1) + 1, video_stride))
    return sorted(steps)


@dataclass
class Chunk:
    start: int
    actions: np.ndarray  # (chunk_len, D) as predicted
    video: Optional[List[Image.Image]]  # frame j predicts recorded step start + j * video_stride
    seconds: float


@dataclass
class Rollout:
    episode: Episode
    horizon: int
    chunk_len: int
    chunks: List[Chunk]

    @property
    def starts(self) -> List[int]:
        return [chunk.start for chunk in self.chunks]

    def executed(self) -> np.ndarray:
        """The first ``horizon`` actions of every chunk, one per recorded step."""
        out = np.empty_like(self.episode.actions)
        for chunk in self.chunks:
            end = min(chunk.start + self.horizon, len(self.episode))
            out[chunk.start : end] = chunk.actions[: end - chunk.start]
        return out

    def hold(self) -> np.ndarray:
        """The trivial predictor: every chunk holds the state it starts from."""
        out = np.empty_like(self.episode.actions)
        for start in self.starts:
            out[start : start + self.horizon] = self.episode.states[start]
        return out


def rollout_episode(policy: ChunkPolicy, episode: Episode, horizon: int) -> Rollout:
    """Predict one chunk per ``horizon`` steps, each from the recorded observation at its start."""
    chunks = []
    for start in chunk_starts(len(episode), horizon):
        began = time.perf_counter()
        actions, video = policy.predict(episode.images(start), episode.states[start], episode.prompt)
        seconds = time.perf_counter() - began
        actions = np.asarray(actions, dtype=np.float64)
        expected = (policy.chunk_len, episode.actions.shape[1])
        if actions.shape != expected:
            raise ValueError(f"policy returned actions of shape {actions.shape}, expected {expected}")
        chunks.append(Chunk(start=start, actions=actions, video=video, seconds=seconds))
    return Rollout(episode=episode, horizon=horizon, chunk_len=policy.chunk_len, chunks=chunks)


def dimension_groups(names: Sequence[str]) -> Dict[str, List[int]]:
    """Grippers in one group, the other dimensions by their ``left_`` / ``right_`` prefix."""
    groups: Dict[str, List[int]] = {}
    for i, name in enumerate(names):
        if "gripper" in name:
            key = "grippers"
        elif name.startswith(("left_", "right_")):
            key = f"{name.split('_')[0]} joints"
        else:
            key = "joints"
        groups.setdefault(key, []).append(i)
    return groups


def chunk_step_errors(rollout: Rollout) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summed ``|error|`` at each in-chunk step over every chunk, for the policy and for holding.

    Covers all ``chunk_len`` predicted steps, not only the executed ones. Returns
    ``(policy_sums, hold_sums, counts)`` with sums of shape ``(chunk_len, D)``;
    sums rather than means, so episodes pool with the right weights.
    """
    recorded = rollout.episode.actions
    policy_sums = np.zeros((rollout.chunk_len, recorded.shape[1]))
    hold_sums = np.zeros_like(policy_sums)
    counts = np.zeros(rollout.chunk_len)
    for chunk in rollout.chunks:
        k = min(rollout.chunk_len, len(recorded) - chunk.start)
        target = recorded[chunk.start : chunk.start + k]
        policy_sums[:k] += np.abs(chunk.actions[:k] - target)
        hold_sums[:k] += np.abs(rollout.episode.states[chunk.start] - target)
        counts[:k] += 1
    return policy_sums, hold_sums, counts


def psnr(generated: Image.Image, reference: Image.Image) -> float:
    if reference.size != generated.size:
        reference = reference.resize(generated.size, Image.BILINEAR)
    a = np.asarray(generated.convert("RGB"), dtype=np.float64)
    b = np.asarray(reference.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    return MAX_PSNR_DB if mse == 0 else min(MAX_PSNR_DB, 10.0 * math.log10(255.0**2 / mse))


@dataclass
class VideoFrame:
    chunk: int
    offset: int  # generated frame index j within its chunk
    step: int  # recorded step it corresponds to
    generated: Image.Image
    recorded: Optional[Image.Image]  # None past the end of the episode
    psnr: Optional[float]


def video_frames(rollout: Rollout, policy: ChunkPolicy) -> List[VideoFrame]:
    """Pair every generated frame with the recorded canvas at the step it stands for."""
    episode, stride = rollout.episode, policy.video_stride
    canvases: Dict[int, Image.Image] = {}
    out = []
    for ci, chunk in enumerate(rollout.chunks):
        for j, generated in enumerate(chunk.video or []):
            step = chunk.start + j * stride
            recorded = None
            if step < len(episode):
                if step not in canvases:
                    canvases[step] = policy.canvas(episode.images(step), episode.states[step])
                recorded = canvases[step]
            score = psnr(generated, recorded) if recorded is not None else None
            out.append(VideoFrame(ci, j, step, generated, recorded, score))
    return out


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _finite(values: np.ndarray) -> List[Optional[float]]:
    """NaN (an in-chunk step no episode reached) as None, so summary.json stays standard JSON."""
    return [None if math.isnan(v) else float(v) for v in values]


def evaluate(
    policy: ChunkPolicy,
    dataset_dir: str | Path,
    episodes: Sequence[int],
    out_dir: str | Path,
    *,
    horizon: Optional[int] = None,
    video: bool = True,
    metadata: Optional[dict] = None,
) -> dict:
    """Evaluate ``episodes`` and write per-episode plots, videos and arrays plus a summary.

    Writes ``episode_XXXXXX_actions.png``, ``episode_XXXXXX_video.mp4`` (when
    ``video``), ``episode_XXXXXX.npz`` and ``summary.json`` / ``summary.png``
    into ``out_dir``. Returns the summary.
    """
    from openwam.eval import visualize

    horizon = int(horizon or policy.chunk_len)
    if not 0 < horizon <= policy.chunk_len:
        raise ValueError(f"horizon must be in [1, {policy.chunk_len}], got {horizon}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names: List[str] = []
    per_episode, all_policy_err, all_hold_err = [], [], []
    sums = None
    psnr_by_offset: Dict[int, List[float]] = {}
    for index in episodes:
        episode = read_episode(dataset_dir, index, policy.cameras)
        names = episode.names
        episode.load_frames(
            frame_steps(len(episode), horizon, policy.chunk_len, policy.video_stride if video else None)
        )
        rollout = rollout_episode(policy, episode, horizon)
        executed, hold = rollout.executed(), rollout.hold()
        policy_err, hold_err = np.abs(executed - episode.actions), np.abs(hold - episode.actions)
        all_policy_err.append(policy_err)
        all_hold_err.append(hold_err)
        step_sums = chunk_step_errors(rollout)
        sums = step_sums if sums is None else tuple(a + b for a, b in zip(sums, step_sums))

        stem = f"episode_{index:06d}"
        groups = dimension_groups(names)
        group_mae = {g: float(policy_err[:, idx].mean()) for g, idx in groups.items()}
        visualize.plot_action_trajectories(
            episode.actions,
            executed,
            names,
            rollout.starts,
            out / f"{stem}_actions.png",
            title=f"episode {index}: " + ", ".join(f"{g} MAE {v:.4f}" for g, v in group_mae.items()),
            fps=episode.fps,
        )
        frames = video_frames(rollout, policy) if video else []
        if frames:
            visualize.write_comparison_video(
                frames, len(rollout.chunks), out / f"{stem}_video.mp4", fps=episode.fps / policy.video_stride
            )
        for frame in frames:
            if frame.psnr is not None:
                psnr_by_offset.setdefault(frame.offset, []).append(frame.psnr)
        np.savez_compressed(
            out / f"{stem}.npz",
            recorded=episode.actions,
            executed=executed,
            hold=hold,
            chunk_starts=np.asarray(rollout.starts),
            chunk_actions=np.stack([chunk.actions for chunk in rollout.chunks]),
        )
        predicted = [f.psnr for f in frames if f.offset > 0]
        result = {
            "episode": index,
            "steps": len(episode),
            "chunks": len(rollout.chunks),
            "mae": group_mae,
            "hold_mae": {g: float(hold_err[:, idx].mean()) for g, idx in groups.items()},
            "mae_per_dim": dict(zip(names, policy_err.mean(axis=0).tolist())),
            "max_abs_err_per_dim": dict(zip(names, policy_err.max(axis=0).tolist())),
            "video_psnr": _mean(predicted),
            "recon_psnr": _mean(f.psnr for f in frames if f.offset == 0),
            "seconds_per_chunk": float(np.mean([chunk.seconds for chunk in rollout.chunks])),
        }
        per_episode.append(result)
        logger.info(
            "episode %d: %d steps, %d chunks | %s | video PSNR %s dB",
            index,
            result["steps"],
            result["chunks"],
            ", ".join(f"{g} MAE {v:.4f} (hold {result['hold_mae'][g]:.4f})" for g, v in group_mae.items()),
            "n/a" if result["video_psnr"] is None else f"{result['video_psnr']:.2f}",
        )

    policy_err, hold_err = np.concatenate(all_policy_err), np.concatenate(all_hold_err)
    policy_sums, hold_sums, counts = sums
    with np.errstate(invalid="ignore"):
        by_step, hold_by_step = policy_sums / counts[:, None], hold_sums / counts[:, None]
    groups = dimension_groups(names)
    summary = {
        **(metadata or {}),
        "dataset_dir": str(dataset_dir),
        "episodes": list(episodes),
        "horizon": horizon,
        "dimension_names": names,
        "overall": {
            "mae": {g: float(policy_err[:, idx].mean()) for g, idx in groups.items()},
            "hold_mae": {g: float(hold_err[:, idx].mean()) for g, idx in groups.items()},
            "mae_per_dim": dict(zip(names, policy_err.mean(axis=0).tolist())),
            "mae_by_chunk_step": {g: _finite(by_step[:, idx].mean(axis=1)) for g, idx in groups.items()},
            "hold_mae_by_chunk_step": {g: _finite(hold_by_step[:, idx].mean(axis=1)) for g, idx in groups.items()},
            "video_psnr_by_offset": {
                int(j) * policy.video_stride: float(np.mean(v)) for j, v in sorted(psnr_by_offset.items())
            },
        },
        "per_episode": per_episode,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    visualize.plot_summary(summary, out / "summary.png")
    return summary


__all__ = [
    "ChunkPolicy",
    "Chunk",
    "Episode",
    "Rollout",
    "VideoFrame",
    "chunk_starts",
    "chunk_step_errors",
    "dimension_groups",
    "evaluate",
    "frame_steps",
    "psnr",
    "read_episode",
    "rollout_episode",
    "video_frames",
]
