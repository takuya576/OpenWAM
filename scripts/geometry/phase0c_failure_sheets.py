#!/usr/bin/env python3
"""Contact sheets of closed-loop YAM rollouts, for labelling why failures failed.

For each ``ep<N>_imagined_vs_live.mp4`` (scripts/run_yam_policy.py) selected,
one JPEG: the live canvas at every chunk start, then the last live canvas and
the model's imagined last frame of the final chunk. Tiles are labelled with
the chunk number. Failures are always included; ``--successes N`` adds the
first N successful episodes of each directory as references.

    python scripts/geometry/phase0c_failure_sheets.py \\
        --dirs outputs/eval_out/hang_ring_n50_delta_lora/*/checkpoint_step_5120/closed_loop \\
        --out outputs/geometry_phase0/failures
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W, GAP, PER_CHUNK = 320, 2, 9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True, help="closed_loop directories (globs allowed)")
    p.add_argument("--out", required=True)
    p.add_argument("--successes", type=int, default=2)
    p.add_argument("--scale", type=float, default=0.75)
    p.add_argument("--per-row", type=int, default=8)
    return p.parse_args()


def task_name(d: Path) -> str:
    return d.parts[d.parts.index("eval_out") + 1]


def sheet(video: Path, scale: float, per_row: int, title: str) -> Image.Image:
    import av

    starts, last_live, last_imagined = [], None, None
    with av.open(str(video)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    n_chunks = len(frames) // PER_CHUNK
    for i, a in enumerate(frames):
        chunk, j = divmod(i, PER_CHUNK)
        live = a[:, PANEL_W + GAP : 2 * PANEL_W + GAP]
        executed = float(a[45:, 2 * (PANEL_W + GAP) :].mean()) > 0.3
        if j == 0:
            starts.append((chunk, live))
        if executed:
            last_live = (chunk, j, live)
        if chunk == n_chunks - 1 and j == PER_CHUNK - 1:
            last_imagined = a[:, :PANEL_W]
    font = ImageFont.load_default(size=16)
    tiles = [(f"c{c}", img) for c, img in starts]
    if last_live is not None:
        tiles.append((f"final live c{last_live[0]} f{last_live[1]}", last_live[2]))
    if last_imagined is not None:
        tiles.append(("imagined end of last chunk", last_imagined))
    tw, th = int(PANEL_W * scale), int(384 * scale)
    rows = (len(tiles) + per_row - 1) // per_row
    out = Image.new("RGB", (per_row * (tw + 4), rows * (th + 4) + 24), "white")
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), title, fill="black", font=font)
    for k, (label, img) in enumerate(tiles):
        r, col = divmod(k, per_row)
        x, y = col * (tw + 4), 24 + r * (th + 4)
        out.paste(Image.fromarray(np.ascontiguousarray(img)).resize((tw, th), Image.Resampling.BILINEAR), (x, y))
        draw.text((x + 4, y + th - 20), label, fill="yellow", font=font, stroke_width=2, stroke_fill="black")
    return out


def main():
    args = parse_args()
    dirs = sorted({Path(p) for g in args.dirs for p in glob.glob(g)})
    for d in dirs:
        task = task_name(d)
        ckpt = d.parent.name
        summaries = []
        for f in d.glob("ep*_summary.json"):
            s = json.loads(f.read_text())
            summaries.append((int(re.match(r"ep(\d+)_", f.name).group(1)), s.get("success"), s.get("chunks")))
        summaries.sort()
        chosen = [s for s in summaries if s[1] is False]
        chosen += [s for s in summaries if s[1] is True][: args.successes]
        out_dir = Path(args.out) / task
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep, success, chunks in chosen:
            video = d / f"ep{ep}_imagined_vs_live.mp4"
            if not video.exists():
                continue
            tag = "ok" if success else "fail"
            title = f"{task} {ckpt} ep{ep} {'SUCCESS' if success else 'FAILURE'} ({chunks} chunks)"
            sheet(video, args.scale, args.per_row, title).save(out_dir / f"ep{ep:03d}_{tag}.jpg", quality=88)
        print(f"{task}/{ckpt}: {sum(1 for c in chosen if c[1] is False)} failures + "
              f"{sum(1 for c in chosen if c[1])} successes -> {out_dir}")


if __name__ == "__main__":
    main()
