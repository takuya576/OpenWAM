#!/usr/bin/env python3
"""Phase 0C: is the geometry of OpenWAM's *imagined* future frames right?

Reads the comparison videos the evaluation scripts already wrote — each frame is
``imagined | real | |difference|`` (three 320x384 canvases, 2-px gaps), 9 frames
per chunk, frame 0 being the VAE reconstruction of the observation:

* open loop  (scripts/open_loop_eval.py, ``episode_*_video.mp4``): real = the
  recorded held-out episode at the step the frame stands for;
* closed loop (scripts/run_yam_policy.py, ``ep*_imagined_vs_live.mp4``): real =
  the live robot canvas, with the episode's success label from ``ep*_summary.json``.

Every camera slot of both canvases is un-stretched to 4:3 (``--da3-hw``) and
given to DA3-Giant as its own one-view set. Per (frame, camera) it records:

* abs_rel, edge_f1            : imagined vs real depth (median-scaled AbsRel;
                                |grad log depth| > tau edges matched within tol px)
* hold_abs_rel, hold_edge_f1  : the chunk's real frame 0 vs real frame j — a
                                model that predicted no motion
* hold_vae_*                  : the same no-motion model, but using the imagined
                                frame 0 (the Wan VAE reconstruction of the observation),
                                so it carries the same VAE blur as the imagined frames
* *_dyn                       : the same, only where the real scene moved since frame 0
                                (|Δ log depth| > 0.05 after removing each map's median
                                log depth, dilated 4 px), i.e. arm / object
* dyn_frac                    : that region's share of the image
* psnr                        : imagined vs real appearance (control)

Frames whose difference panel is blank (past the episode end, or not executed
in closed loop) are skipped; the burnt-in labels in the head slot are masked.

Run in the ``gae`` conda env:
    CUDA_VISIBLE_DEVICES=4 python scripts/geometry/phase0c_imagined_geometry.py \\
        --videos 'outputs/eval_out/hang_ring_n50_delta_lora/*/checkpoint_step_5120/closed_loop/ep*_imagined_vs_live.mp4' \\
        --out outputs/geometry_phase0/imagined/hang_ring_closed.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gae_tools as G  # noqa: E402

PANEL_W, GAP, FRAMES_PER_CHUNK = 320, 2, 9
# Burnt-in text (two lines at y=4 / y=22, ~13 px font) in the canvas' top-left = head slot.
LABEL_BOX = (42, 250)  # canvas rows, cols to ignore in the head slot


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, nargs="+", help="glob(s) of comparison videos")
    p.add_argument("--out", required=True, help="per-frame CSV")
    p.add_argument("--da3-hw", type=int, nargs=2, default=(252, 336))
    p.add_argument("--tau", type=float, default=0.03)
    p.add_argument("--tol", type=int, default=2)
    p.add_argument("--dyn-thresh", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def episode_meta(path: Path) -> dict:
    """Mode, episode id and (closed loop) success label of one comparison video."""
    m = re.match(r"ep(\d+)_imagined_vs_live\.mp4", path.name)
    if m:
        summary = json.loads((path.parent / f"ep{m.group(1)}_summary.json").read_text())
        return {"mode": "closed_loop", "episode": int(m.group(1)), "success": summary.get("success")}
    m = re.match(r"episode_(\d+)_video\.mp4", path.name)
    if m:
        return {"mode": "open_loop", "episode": int(m.group(1)), "success": None}
    raise ValueError(f"unrecognized video name {path.name}")


def read_panels(path: Path):
    """Yield (frame index, imagined canvas, real canvas, executed) from a comparison video."""
    import av

    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            a = frame.to_ndarray(format="rgb24")
            imagined = a[:, :PANEL_W]
            real = a[:, PANEL_W + GAP : 2 * PANEL_W + GAP]
            diff = a[45:, 2 * (PANEL_W + GAP) :]
            yield i, imagined, real, float(diff.mean()) > 0.3


def to_input(images, hw) -> torch.Tensor:
    h, w = hw
    arr = [np.asarray(Image.fromarray(x).resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) for x in images]
    return torch.from_numpy(np.stack(arr) / 255.0).permute(0, 3, 1, 2).contiguous()


def masked_agreement(pred, ref, valid) -> float:
    if valid.sum() < 50:
        return float("nan")
    p, r = pred[valid].clamp_min(1e-4), ref[valid].clamp_min(1e-4)
    p = p * (r.median() / p.median())
    return float(((p - r).abs() / r).mean())


def masked_f1(pred_e, ref_e, valid, tol) -> float:
    pe, re_ = pred_e & valid, ref_e & valid
    m = G.edge_f1(pe, re_, tol)
    return m["f1"]


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    args = parse_args()
    paths = sorted({Path(p) for g in args.videos for p in glob.glob(g)}, key=lambda p: (str(p.parent), episode_meta(p)["episode"]))
    if not paths:
        raise SystemExit(f"no videos match {args.videos}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = G.load_gae(device)
    hw = tuple(args.da3_hw)
    lab_h = int(round(LABEL_BOX[0] * hw[0] / 256))
    lab_w = int(round(LABEL_BOX[1] * hw[1] / 320))
    cams = [G.CAMERA_SHORT[c] for c in G.CAMERAS]

    fields = ["video", "mode", "episode", "success", "chunk", "j", "camera", "abs_rel", "edge_f1", "hold_abs_rel",
              "hold_edge_f1", "hold_vae_abs_rel", "hold_vae_edge_f1", "abs_rel_dyn", "edge_f1_dyn",
              "hold_abs_rel_dyn", "hold_edge_f1_dyn", "hold_vae_abs_rel_dyn", "hold_vae_edge_f1_dyn", "dyn_frac", "psnr"]
    t0 = time.time()
    n_rows = 0
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for vi, path in enumerate(paths):
            meta = episode_meta(path)
            frames = [(i, im, re_) for i, im, re_, executed in read_panels(path) if executed]
            if not frames:
                continue
            # DA3 on every slot of every kept imagined and real canvas.
            images, index = [], []
            for i, im, re_ in frames:
                for kind, canvas in (("imagined", im), ("real", re_)):
                    for cam, img in G.canvas_slots(canvas).items():
                        images.append(img)
                        index.append((i, kind, cam))
            depth = []
            for b in range(0, len(images), args.batch):
                x = to_input(images[b : b + args.batch], hw).to(device)
                depth.append(G.da3_depth(model, x))
            depth = torch.cat(depth)
            lookup = {key: depth[k] for k, key in enumerate(index)}
            edges = {key: G.log_depth_gradient(depth[k][None])[0] > args.tau for k, key in enumerate(index)}
            slot_img = {key: images[k] for k, key in enumerate(index)}

            def rel_log(d):
                ld = torch.log(d.clamp_min(1e-4))
                return ld - ld.median()

            full = torch.ones(hw, dtype=torch.bool, device=device)
            valid_head = full.clone()
            valid_head[:lab_h, :lab_w] = False
            for i, _, _ in frames:
                chunk, j = divmod(i, FRAMES_PER_CHUNK)
                ref0 = chunk * FRAMES_PER_CHUNK  # frame 0 of this chunk
                for cam in cams:
                    valid = valid_head if cam == "head" else full
                    k_im, k_re = (i, "imagined", cam), (i, "real", cam)
                    d_im, d_re = lookup[k_im], lookup[k_re]
                    row = {"video": str(path), **meta, "chunk": chunk, "j": j, "camera": cam,
                           "abs_rel": masked_agreement(d_im, d_re, valid),
                           "edge_f1": masked_f1(edges[k_im], edges[k_re], valid, args.tol),
                           "psnr": psnr_u8(slot_img[k_im], slot_img[k_re])}
                    k_0, k_v0 = (ref0, "real", cam), (ref0, "imagined", cam)
                    if j > 0 and k_0 in lookup:
                        d_0, d_v0 = lookup[k_0], lookup[k_v0]
                        moved = (rel_log(d_re) - rel_log(d_0)).abs() > args.dyn_thresh
                        dyn = G.dilate(moved[None], 4)[0] & valid
                        row.update({
                            "hold_abs_rel": masked_agreement(d_0, d_re, valid),
                            "hold_edge_f1": masked_f1(edges[k_0], edges[k_re], valid, args.tol),
                            "hold_vae_abs_rel": masked_agreement(d_v0, d_re, valid),
                            "hold_vae_edge_f1": masked_f1(edges[k_v0], edges[k_re], valid, args.tol),
                            "abs_rel_dyn": masked_agreement(d_im, d_re, dyn),
                            "edge_f1_dyn": masked_f1(edges[k_im], edges[k_re], dyn, args.tol),
                            "hold_abs_rel_dyn": masked_agreement(d_0, d_re, dyn),
                            "hold_edge_f1_dyn": masked_f1(edges[k_0], edges[k_re], dyn, args.tol),
                            "hold_vae_abs_rel_dyn": masked_agreement(d_v0, d_re, dyn),
                            "hold_vae_edge_f1_dyn": masked_f1(edges[k_v0], edges[k_re], dyn, args.tol),
                            "dyn_frac": float(dyn.float().mean()),
                        })
                    writer.writerow(row)
                    n_rows += 1
            f.flush()
            print(f"[{vi + 1}/{len(paths)}] {path.parent.parent.name}/{path.name}: {len(frames)} frames "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {out} ({n_rows} rows)")


if __name__ == "__main__":
    main()
