#!/usr/bin/env python3
"""GAE target cache for the alignment loss (``training.gae_align``).

Encodes every frame of every camera of a YAM LeRobot v2.1 dataset with the frozen
DA3-Giant + GAE-64 codec and stores the posterior mean, standardized with GAE's
latent statistics:

    <out>/episode_000012.npy   float16, (T, 3, 64, 27, 36)   cameras in gae_tools.CAMERAS order
    <out>/meta.json            cameras, grid, input size, dataset, per-episode lengths

Each frame is encoded on its own (a one-view set) from the ORIGINAL 640x480 camera
image resized to 378x504 — not from the 384x320 training canvas — so a wrist view
keeps 27x36 geometry tokens although it occupies only 4x5 DiT tokens in the canvas.

Run in the ``gae`` conda env.  Episodes can be split over GPUs with --shard i/n:

    CUDA_VISIBLE_DEVICES=0 python scripts/geometry/gae_cache.py \\
        --dataset-dir assets/datasets/hang_ring --out assets/gae_cache/hang_ring --shard 0/2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gae_tools as G  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default=None, help="half-open range 'a:b'; default all")
    p.add_argument("--shard", default="0/1", help="i/n: take every n-th episode starting at i")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lengths = G.episode_lengths(args.dataset_dir)
    episodes = sorted(lengths) if args.episodes is None else G.parse_range(args.episodes)
    i, n = (int(x) for x in args.shard.split("/"))
    mine = episodes[i::n]

    meta = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "cameras": list(G.CAMERAS),
        "channels": 64,
        "grid": [G.DA3_H // 14, G.DA3_W // 14],
        "input_size": [G.DA3_H, G.DA3_W],
        "encoder": "DA3-Giant + GAE-64 posterior mean, standardized (latent_stats_gae_64.pt), single-frame",
        "dtype": "float16",
        "lengths": {str(e): int(lengths[e]) for e in episodes},
    }
    if i == 0:
        (out / "meta.json").write_text(json.dumps(meta, indent=2))

    device = torch.device(args.device)
    model = G.load_gae(device)
    t0 = time.time()
    done = 0
    for ep in mine:
        path = out / f"episode_{ep:06d}.npy"
        length = lengths[ep]
        if path.exists():
            try:
                if np.load(path, mmap_mode="r").shape[0] == length:
                    continue
            except Exception:
                pass
        buf = np.zeros((length, len(G.CAMERAS), 64, G.DA3_H // 14, G.DA3_W // 14), dtype=np.float16)
        for c, cam in enumerate(G.CAMERAS):
            frames = G.decode_frames(G.video_path(args.dataset_dir, cam, ep), range(length))
            for b in range(0, length, args.batch):
                idx = list(range(b, min(b + args.batch, length)))
                x = G.to_da3_input([frames[k] for k in idx]).to(device)
                _, mu = G.encode(model, x[:, None])
                buf[idx, c] = model.standardize(mu).float().cpu().numpy().astype(np.float16)
            del frames
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, buf)
        tmp.rename(path)
        done += 1
        print(
            f"episode {ep}: {length} frames x {len(G.CAMERAS)} cameras, |z| mean {np.abs(buf).mean():.3f}, "
            f"{done}/{len(mine)} done, {time.time() - t0:.0f}s",
            flush=True,
        )
    print("finished", flush=True)


if __name__ == "__main__":
    main()
