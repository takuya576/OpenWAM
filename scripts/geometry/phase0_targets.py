#!/usr/bin/env python3
"""Phase 0B (step 1): geometry targets for probing OpenWAM's frame-0 tokens.

Picks window start frames from one YAM task, encodes every camera frame alone
with DA3-Giant + GAE (exactly what a policy could compute live), and writes
the targets in OpenWAM's 384x320 canvas layout so they line up with the DiT's
token grid (12x10 tokens of 32 px):

* depth_edge (N, 48, 40) : max |grad log depth| of the DA3 teacher depth per 8-px cell
                           (4x4 cells per token; threshold it with tau later)
* logdepth   (N, 48, 40) : mean log depth per 8-px cell minus its camera slot's median
* gae        (N, 64, 24, 20): GAE posterior mean, standardized with GAE's latent
                           statistics, averaged per 16-px cell (2x2 cells per token) —
                           the same layout Phase 1 would cache
* frames     (N, 3)      : [split code, episode, start frame]; split codes are in
                           ``split_names`` (fit / select / test)

The OpenWAM side (phase0_probe.py) reads ``frames`` to build the identical
dataset windows, so frame selection lives only here.

Run in the ``gae`` conda env:
    CUDA_VISIBLE_DEVICES=4 python scripts/geometry/phase0_targets.py \\
        --dataset-dir assets/datasets/place_cube_right \\
        --fit 0:48 --select 48:56 --test 56:66 \\
        --out outputs/geometry_phase0/place_cube_right/targets.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gae_tools as G  # noqa: E402

SPLIT_NAMES = ("fit", "select", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--fit", required=True, help="episodes the probes are fitted on (from the train split)")
    p.add_argument("--select", required=True, help="train-split episodes used to pick the ridge penalty")
    p.add_argument("--test", required=True, help="held-out episodes (the checkpoint's val split)")
    p.add_argument("--k-train", type=int, default=24, help="window starts per fit/select episode")
    p.add_argument("--k-test", type=int, default=48, help="window starts per test episode")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--debug-frames", type=int, default=4, help="canvas visualizations for the first N test frames")
    p.add_argument("--debug-tau", type=float, default=0.03)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def window_starts(length: int, k: int) -> list[int]:
    last = length - G.WINDOW_FRAMES
    if last < 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(0, last, k)})


def stretch_canvas(frames_by_cam: dict) -> np.ndarray:
    """The RGB canvas as OpenWAM builds it (each camera bilinear-stretched into its slot)."""
    canvas = np.zeros((G.CANVAS_H, G.CANVAS_W, 3), dtype=np.uint8)
    for cam, (top, left, h, w) in G.SLOTS.items():
        canvas[top : top + h, left : left + w] = np.asarray(
            Image.fromarray(frames_by_cam[cam]).resize((w, h), Image.Resampling.BILINEAR)
        )
    return canvas


def save_debug(path: Path, canvas: np.ndarray, edge: np.ndarray, logd: np.ndarray, gae: np.ndarray, tau: float):
    """canvas | depth-edge cells (8 px) | relative log depth (8 px) | GAE latent PCA (16 px)."""
    import matplotlib.cm as cm

    def up(a):
        return np.asarray(Image.fromarray(a).resize((G.CANVAS_W, G.CANVAS_H), Image.Resampling.NEAREST))

    e = up(((edge > tau) * 255).astype(np.uint8))
    e = np.stack([e] * 3, -1)
    ld = np.clip((-logd + 0.5), 0, 1)  # near = bright
    ld = up((cm.viridis(ld)[..., :3] * 255).astype(np.uint8))
    flat = gae.reshape(gae.shape[0], -1).T.astype(np.float32)
    flat = flat - flat.mean(0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    pc = flat @ vt[:3].T
    pc = (pc - pc.min(0)) / (np.ptp(pc, 0) + 1e-6)
    pc = up((pc.reshape(gae.shape[1], gae.shape[2], 3) * 255).astype(np.uint8))
    Image.fromarray(np.concatenate([canvas, e, ld, pc], axis=1)).save(path)


def main():
    args = parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = out.parent / "targets_debug"
    debug_dir.mkdir(exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve()
    device = torch.device(args.device)
    lengths = G.episode_lengths(dataset_dir)

    plan = []  # (split code, episode, starts)
    for code, (spec, k) in enumerate(((args.fit, args.k_train), (args.select, args.k_train), (args.test, args.k_test))):
        for ep in G.parse_range(spec):
            starts = window_starts(lengths[ep], k)
            if starts:
                plan.append((code, ep, starts))
    n_total = sum(len(s) for _, _, s in plan)
    print(f"{n_total} frames from {len(plan)} episodes", flush=True)

    model = G.load_gae(device)
    frames_out, edge_out, logd_out, gae_out = [], [], [], []
    n_debug = 0
    t0 = time.time()
    for code, ep, starts in plan:
        decoded = {cam: G.decode_frames(G.video_path(dataset_dir, cam, ep), starts) for cam in G.CAMERAS}
        per_cam = {}
        for cam in G.CAMERAS:
            imgs = G.to_da3_input([decoded[cam][s] for s in starts])
            depths, mus = [], []
            for i in range(0, len(starts), args.batch):
                x = imgs[i : i + args.batch].to(device).unsqueeze(1)  # one-view sets
                feats, mu = G.encode(model, x)
                depths.append(G.raw_depth(model, feats, G.DA3_H, G.DA3_W))
                mus.append(model.standardize(mu))
            depth = torch.cat(depths)
            per_cam[cam] = {
                "edge": G.log_depth_gradient(depth).unsqueeze(1),
                "logd": torch.log(depth.clamp_min(1e-4)).unsqueeze(1),
                "gae": torch.cat(mus),
            }
        edge = G.to_canvas({c: v["edge"] for c, v in per_cam.items()}, cell=8, reduce="max")[:, 0]
        logd = G.to_canvas({c: v["logd"] for c, v in per_cam.items()}, cell=8, reduce="avg")[:, 0]
        for cam, mask in zip(G.CAMERAS, G.slot_masks(8).values()):
            m = torch.from_numpy(mask).to(logd.device)
            med = logd[:, m].median(dim=1).values
            logd[:, m] -= med.unsqueeze(1)
        gae = G.to_canvas({c: v["gae"] for c, v in per_cam.items()}, cell=16, reduce="avg")

        frames_out.append(np.array([[code, ep, s] for s in starts], dtype=np.int64))
        edge_out.append(edge.cpu().numpy().astype(np.float16))
        logd_out.append(logd.cpu().numpy().astype(np.float16))
        gae_out.append(gae.cpu().numpy().astype(np.float16))

        if code == 2 and n_debug < args.debug_frames:
            for j, s in enumerate(starts[: args.debug_frames - n_debug]):
                canvas = stretch_canvas({cam: decoded[cam][s] for cam in G.CAMERAS})
                save_debug(
                    debug_dir / f"ep{ep:03d}_f{s:04d}.png",
                    canvas, edge_out[-1][j].astype(np.float32), logd_out[-1][j].astype(np.float32),
                    gae_out[-1][j].astype(np.float32), args.debug_tau,
                )
                n_debug += 1
        done = sum(len(f) for f in frames_out)
        print(f"[{SPLIT_NAMES[code]} ep {ep}] {done}/{n_total} frames ({time.time() - t0:.0f}s)", flush=True)

    meta = {
        "dataset_dir": str(dataset_dir),
        "splits": {"fit": args.fit, "select": args.select, "test": args.test},
        "k_train": args.k_train,
        "k_test": args.k_test,
        "window_frames": G.WINDOW_FRAMES,
        "cameras": list(G.CAMERAS),
        "slots": {G.CAMERA_SHORT[c]: v for c, v in G.SLOTS.items()},
        "da3_input_hw": [G.DA3_H, G.DA3_W],
        "encoding": "single frame per camera (one-view DA3 set)",
        "gae": "GAE-D64 gae_64.pt posterior mean, standardized (latent_stats_gae_64.pt)",
    }
    np.savez_compressed(
        out,
        frames=np.concatenate(frames_out),
        depth_edge=np.concatenate(edge_out),
        logdepth=np.concatenate(logd_out),
        gae=np.concatenate(gae_out),
        split_names=np.array(SPLIT_NAMES),
        meta=np.array(json.dumps(meta)),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
