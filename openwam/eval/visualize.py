"""Plots and the side-by-side video for open-loop evaluation.

matplotlib is imported only when plotting (the ``eval`` extra); the video needs
just Pillow and imageio's bundled ffmpeg.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# The difference panel scales |generated - recorded| by this so small errors stay visible.
DIFF_GAIN = 3
_GAP = 2  # white divider between panels, in pixels


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError("open-loop plots need matplotlib: pip install -e '.[eval]'") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _columns(names: Sequence[str]) -> List[List[int]]:
    """Dimension indices per plot column: one column per arm when names split into left_/right_."""
    left = [i for i, n in enumerate(names) if n.startswith("left_")]
    right = [i for i, n in enumerate(names) if n.startswith("right_")]
    if left and len(left) == len(right) and len(left) + len(right) == len(names):
        return [left, right]
    rows = math.ceil(len(names) / 2)
    return [list(range(rows)), list(range(rows, len(names)))]


def plot_action_trajectories(
    recorded: np.ndarray,
    executed: np.ndarray,
    names: Sequence[str],
    chunk_starts: Sequence[int],
    path: str | Path,
    *,
    title: str,
    fps: float,
) -> None:
    """Recorded vs predicted actions, one panel per dimension, chunk starts marked."""
    plt = _pyplot()
    columns = _columns(names)
    nrows = max(len(column) for column in columns)
    fig, axes = plt.subplots(
        nrows,
        len(columns),
        figsize=(7.5 * len(columns), 1.7 * nrows + 1),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    steps = np.arange(len(recorded))
    for col, dims in enumerate(columns):
        for row in range(nrows):
            ax = axes[row, col]
            if row >= len(dims):
                ax.axis("off")
                continue
            d = dims[row]
            ax.plot(steps, recorded[:, d], color="black", lw=1.4, label="recorded")
            ax.plot(steps, executed[:, d], color="tab:orange", lw=1.4, ls="--", label="predicted")
            for start in chunk_starts[1:]:
                ax.axvline(start, color="0.65", lw=0.8, ls=":", zorder=0)
            ax.set_ylabel(names[d], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        axes[-1, col].set_xlabel(f"recorded step ({fps:g} Hz)")
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        f"{title}\ndotted lines: chunk starts (each chunk predicted from the recorded observation)", fontsize=10
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(summary: dict, path: str | Path) -> None:
    """Error against step within the chunk (policy vs holding the start state) and video PSNR by offset."""
    plt = _pyplot()
    overall = summary["overall"]
    groups = list(overall["mae_by_chunk_step"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), layout="constrained")
    for ax, wanted in ((axes[0], [g for g in groups if g != "grippers"]), (axes[1], ["grippers"])):
        for i, group in enumerate(g for g in wanted if g in overall["mae_by_chunk_step"]):
            color = f"C{i}"
            k = np.arange(len(overall["mae_by_chunk_step"][group]))
            ax.plot(k, _floats(overall["mae_by_chunk_step"][group]), color=color, lw=1.6, label=f"{group}: policy")
            ax.plot(
                k,
                _floats(overall["hold_mae_by_chunk_step"][group]),
                color=color,
                lw=1.2,
                ls="--",
                label=f"{group}: hold start state",
            )
        ax.set_xlabel("step within chunk (k)")
        ax.set_ylabel("mean |predicted - recorded|")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_title("joints")
    axes[1].set_title("grippers")
    by_offset = overall["video_psnr_by_offset"]
    if by_offset:
        keys = sorted(by_offset, key=int)  # int keys in memory, str keys once read back from JSON
        offsets, values = [int(k) for k in keys], [by_offset[k] for k in keys]
        axes[2].plot(offsets, values, marker="o", color="tab:green")
        axes[2].annotate(
            "offset 0: VAE reconstruction\nof the input observation",
            (offsets[0], values[0]),
            fontsize=8,
            xytext=(8, -18),
            textcoords="offset points",
        )
    axes[2].set_xlabel("recorded steps after the conditioning observation")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].set_title("generated video vs recorded observation")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"open-loop summary over episodes {summary['episodes']}", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _floats(values) -> List[float]:
    return [math.nan if v is None else v for v in values]


def _label(draw: ImageDraw.ImageDraw, xy, text: str, font) -> None:
    draw.text(xy, text, fill="white", font=font, stroke_width=2, stroke_fill="black")


def comparison_frame(frame, n_chunks: int, font=None) -> np.ndarray:
    """Generated | recorded | amplified |difference| for one :class:`~openwam.eval.open_loop.VideoFrame`."""
    font = font or ImageFont.load_default(size=13)
    generated = frame.generated.convert("RGB")
    w, h = generated.size
    if frame.recorded is not None:
        recorded = frame.recorded.convert("RGB").resize((w, h), Image.BILINEAR)
        delta = np.abs(np.asarray(generated, np.int16) - np.asarray(recorded, np.int16)).mean(axis=2)
        diff = Image.fromarray(np.clip(delta * DIFF_GAIN, 0, 255).astype(np.uint8)).convert("RGB")
    else:
        recorded = diff = Image.new("RGB", (w, h))
    out = Image.new("RGB", (3 * w + 2 * _GAP, h), "white")
    for i, panel in enumerate((generated, recorded, diff)):
        out.paste(panel, (i * (w + _GAP), 0))
    draw = ImageDraw.Draw(out)
    _label(draw, (6, 4), "input, VAE reconstruction" if frame.offset == 0 else "generated", font)
    _label(draw, (6, 22), f"chunk {frame.chunk + 1}/{n_chunks}  frame {frame.offset}  step {frame.step}", font)
    _label(draw, (w + _GAP + 6, 4), "recorded" if frame.recorded is not None else "past episode end", font)
    _label(draw, (2 * (w + _GAP) + 6, 4), f"|difference| x{DIFF_GAIN}", font)
    return np.asarray(out)


def write_comparison_video(frames: Sequence, n_chunks: int, path: str | Path, *, fps: float) -> None:
    """H.264 / yuv420p mp4 (plays in browsers and VS Code) of every generated frame beside the recording."""
    import imageio.v2 as imageio

    font = ImageFont.load_default(size=13)
    writer = imageio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        quality=None,
        ffmpeg_params=["-crf", "20", "-movflags", "+faststart"],
    )
    try:
        for frame in frames:
            writer.append_data(comparison_frame(frame, n_chunks, font))
    finally:
        writer.close()


__all__ = ["comparison_frame", "plot_action_trajectories", "plot_summary", "write_comparison_video"]
