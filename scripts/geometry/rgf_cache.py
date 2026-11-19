#!/usr/bin/env python3
"""Teacher-feature cache for the Relational Geometry Forcing baseline (``training.rgf``).

For every time step of every episode of a YAM LeRobot v2.1 dataset, the three camera views
(head, left wrist, right wrist; original 640x480) are encoded TOGETHER as one multi-view set by a
frozen 3D foundation model, and its final-layer patch features are stored:

    <out>/episode_000012_grid.npy  float16 (T, 120, D)  patches bilinearly resampled (antialiased) to each
                                   view's token slot on the 12x10 canvas grid (head 8x10, wrists 4x5),
                                   in canvas token order (row-major)
    <out>/episode_000012_pool.npy  float16 (T, 3, D)    per-view spatial mean of the full-resolution patches
    <out>/meta.json

Teachers:
    da3   DA3-Giant (the backbone of GAE; depth-anything/DA3-GIANT-1.1, already used by gae_tools.py):
          input 378x504 per view, final block (39 of 40), D = 3072, 27x36 patches.   Run in the `gae` env.
    vggt  VGGT-1B as in the paper: input 518x392 per view, last aggregator layer, D = 2048, 28x37 patches.
          Needs the VGGT source (--vggt-src, a clone of facebookresearch/vggt) and the checkpoint
          (--vggt-weights; default assets/vggt/VGGT-1B/model.safetensors from facebook/VGGT-1B at revision
          860abec7, sha256 f164acf6..., license CC-BY-NC-4.0).
    gae   DA3-Giant + GAE-64, i.e. exactly the targets of the pointwise GAE alignment (training.gae_align): read
          from that run's cache (--gae-cache, written by scripts/geometry/gae_cache.py), D = 64, 27x36 patches.
          Standardized posterior means; each view encoded on its own (single frame), not as a multi-view set.
          No model is run and no video decoded, so it needs no GPU.

    CUDA_VISIBLE_DEVICES=0 python scripts/geometry/rgf_cache.py --teacher da3 \\
        --dataset-dir assets/datasets/hang_ring --out assets/rgf_cache/hang_ring_da3 --shard 0/2
    python scripts/geometry/rgf_cache.py --teacher gae --device cpu \\
        --dataset-dir assets/datasets/hang_ring --out assets/rgf_cache/hang_ring_gae
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gae_tools as G  # noqa: E402

# Slot layout from openwam/model/rgf_align.py, loaded by file path (the teacher env need not import openwam).
_spec = importlib.util.spec_from_file_location("rgf_align", HERE.parents[1] / "openwam" / "model" / "rgf_align.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["rgf_align"] = R  # dataclasses look their module up in sys.modules
_spec.loader.exec_module(R)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", choices=("da3", "vggt", "gae"), default="da3")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", default=None, help="half-open range 'a:b'; default all")
    p.add_argument("--shard", default="0/1", help="i/n: take every n-th episode starting at i")
    p.add_argument("--batch", type=int, default=8, help="time steps per forward (x3 views)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vggt-src", default="/data2/takuya/vggt")
    p.add_argument(
        "--vggt-weights",
        default=str(HERE.parents[1] / "assets" / "vggt" / "VGGT-1B" / "model.safetensors"),
        help="local VGGT-1B checkpoint (.safetensors, or a .pt state dict loaded with weights_only=True)",
    )
    p.add_argument(
        "--gae-cache",
        default=str(HERE.parents[1] / "assets" / "gae_cache" / "hang_ring"),
        help="--teacher gae: the GAE alignment cache (scripts/geometry/gae_cache.py) to read the features from",
    )
    return p.parse_args()


class DA3Teacher:
    name, dim, size = "DA3-Giant (depth-anything/DA3-GIANT-1.1) final block, multi-view set", 3072, (G.DA3_H, G.DA3_W)

    def __init__(self, device):
        self.model = G.load_gae(device)  # GAE wrapper; only its frozen DA3 backbone is used
        self.device = device

    @torch.inference_mode()
    def __call__(self, frames_tv):  # list over time of [head, left, right] HxWx3 uint8 -> (T, V, h, w, D)
        t, v = len(frames_tv), len(frames_tv[0])
        x = G.to_da3_input([f for views in frames_tv for f in views]).to(self.device).reshape(t, v, 3, *self.size)
        bb = self.model.backbone
        x = (x - bb.encoder_mean.to(x).view(1, 1, 3, 1, 1)) / bb.encoder_std.to(x).view(1, 1, 3, 1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = bb.encode(x, mode="all")
        last = feats[max(feats)][:, 1:, :].float()  # drop the extra (class) token
        h, w = self.size[0] // 14, self.size[1] // 14
        if last.shape[1] != h * w:
            raise RuntimeError(f"DA3: {last.shape[1]} patches, expected {h}x{w}")
        return last.reshape(t, v, h, w, -1)


VGGT_HEADS = ("camera_head.", "point_head.", "depth_head.", "track_head.")


def vggt_input_size(width, height, target=518, patch=14):
    """VGGT's own 'crop' preprocessing (vggt/utils/load_fn.py): width -> 518, height rounded to a multiple of 14."""
    new_h = round(height * (target / width) / patch) * patch
    if new_h > target:
        raise ValueError(f"{width}x{height}: VGGT would center-crop the height; not supported here")
    return new_h, target


def load_vggt_aggregator(src, weights):
    """VGGT-1B with only its aggregator (the heads are not needed for the teacher features).

    The checkpoint must provide every aggregator weight; the only keys allowed to be left over are the four heads'.
    """
    sys.path.insert(0, src)
    from vggt.models.vggt import VGGT

    if str(weights).endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(weights)
    else:
        sd = torch.load(weights, map_location="cpu", weights_only=True)
    model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    stray = [k for k in unexpected if not k.startswith(VGGT_HEADS)]
    if missing or stray:
        raise RuntimeError(f"VGGT checkpoint mismatch: missing {missing[:5]}, unexpected non-head keys {stray[:5]}")
    return model, len(sd), len(unexpected)


class VGGTTeacher:
    name, dim = "VGGT-1B (facebook/VGGT-1B) last aggregator layer (frame+global, 2048-d), multi-view set", 2048
    size = vggt_input_size(640, 480)  # (392, 518) for the YAM cameras; checked per frame below

    def __init__(self, device, src, weights):
        if not weights or not Path(weights).is_file():
            raise SystemExit(f"--teacher vggt: no VGGT-1B checkpoint at {weights}")
        model, _, _ = load_vggt_aggregator(src, weights)
        self.model = model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, frames_tv):
        from PIL import Image

        t, v = len(frames_tv), len(frames_tv[0])
        hh, ww = self.size
        for vs in frames_tv:
            for f in vs:
                if vggt_input_size(f.shape[1], f.shape[0]) != self.size:
                    raise ValueError(f"frame {f.shape[1]}x{f.shape[0]} does not map to VGGT input {ww}x{hh}")
        arr = [
            np.asarray(Image.fromarray(f).resize((ww, hh), Image.Resampling.BICUBIC), dtype=np.float32)
            for vs in frames_tv
            for f in vs
        ]
        x = torch.from_numpy(np.stack(arr) / 255.0).permute(0, 3, 1, 2).reshape(t, v, 3, hh, ww).to(self.device)
        # bf16 as in the VGGT README for compute capability >= 8 (A6000 = 8.6); no-op on CPU
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            tokens, start = self.model.aggregator(x)
        last = tokens[-1][:, :, start:].float()  # (t, v, P, 2048); the last layer is always kept by the aggregator
        if last.shape[2] != (hh // 14) * (ww // 14):
            raise RuntimeError(f"VGGT: {last.shape[2]} patches, expected {hh // 14}x{ww // 14}")
        return last.reshape(t, v, hh // 14, ww // 14, -1)


class GAECacheTeacher:
    """Features of the GAE alignment cache: episode_XXXXXX.npy float16 (T, V, 64, 27, 36), cameras in G.CAMERAS order."""

    def __init__(self, cache_dir, dataset_dir):
        self.dir = Path(cache_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        if list(meta["cameras"]) != list(G.CAMERAS):
            raise SystemExit(f"--teacher gae: cache cameras {meta['cameras']} != {list(G.CAMERAS)}")
        if Path(meta["dataset_dir"]).resolve() != Path(dataset_dir).resolve():
            raise SystemExit(f"--teacher gae: cache built from {meta['dataset_dir']}, not {dataset_dir}")
        self.lengths = {int(k): int(v) for k, v in meta["lengths"].items()}
        self.dim, self.size, self.grid = int(meta["channels"]), tuple(meta["input_size"]), tuple(meta["grid"])
        self.name = f"{meta['encoder']} (read from {self.dir})"

    def episode(self, ep, length):  # -> (T, V, h, w, D) float32
        if self.lengths.get(ep) != length:
            raise RuntimeError(f"GAE cache: episode {ep} has {self.lengths.get(ep)} steps, dataset {length}")
        a = np.load(self.dir / f"episode_{ep:06d}.npy", mmap_mode="r")
        if a.shape != (length, len(G.CAMERAS), self.dim, *self.grid):
            raise RuntimeError(f"GAE cache: episode {ep} shape {a.shape}")
        return torch.from_numpy(np.asarray(a, dtype=np.float32)).permute(0, 1, 3, 4, 2)


def to_grid(feat):
    """(T, V, h, w, D) -> canvas-grid targets (T, 120, D) and per-view means (T, V, D)."""
    t, v, h, w, d = feat.shape
    pool = feat.mean(dim=(2, 3))
    grid = feat.new_zeros(t, R.YAM_GRID[0] * R.YAM_GRID[1], d)
    for vi, ((r0, r1, c0, c1), ix) in enumerate(zip(R.YAM_VIEW_SLOTS, R.view_token_indices())):
        x = feat[:, vi].permute(0, 3, 1, 2)  # (T, D, h, w)
        x = F.interpolate(x, size=(r1 - r0, c1 - c0), mode="bilinear", align_corners=False, antialias=True)
        grid[:, ix.to(grid.device)] = x.flatten(2).transpose(1, 2)
    return grid, pool


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lengths = G.episode_lengths(args.dataset_dir)
    episodes = sorted(lengths) if args.episodes is None else G.parse_range(args.episodes)
    i, n = (int(x) for x in args.shard.split("/"))
    mine = episodes[i::n]
    device = torch.device(args.device)
    if args.teacher == "da3":
        teacher = DA3Teacher(device)
    elif args.teacher == "vggt":
        teacher = VGGTTeacher(device, args.vggt_src, args.vggt_weights)
    else:
        teacher = GAECacheTeacher(args.gae_cache, args.dataset_dir)
    meta = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "teacher": args.teacher,
        "teacher_desc": teacher.name,
        "teacher_input": list(teacher.size),
        "dim": teacher.dim,
        "cameras": list(G.CAMERAS),
        "grid": list(R.YAM_GRID),
        "view_slots": [list(s) for s in R.YAM_VIEW_SLOTS],
        "resample": "bilinear, antialias=True, align_corners=False (patch grid -> view slot)",
        "pool": "mean over the full-resolution patch grid",
        "dtype": "float16",
        "lengths": {str(e): int(lengths[e]) for e in episodes},
    }
    if i == 0:
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    t0, done = time.time(), 0
    for ep in mine:
        gpath, ppath = out / f"episode_{ep:06d}_grid.npy", out / f"episode_{ep:06d}_pool.npy"
        length = lengths[ep]
        if gpath.exists() and ppath.exists():
            try:
                if (
                    np.load(gpath, mmap_mode="r").shape[0] == length
                    and np.load(ppath, mmap_mode="r").shape[0] == length
                ):
                    continue
            except Exception:
                pass
        gbuf = np.zeros((length, R.YAM_GRID[0] * R.YAM_GRID[1], teacher.dim), dtype=np.float16)
        pbuf = np.zeros((length, len(G.CAMERAS), teacher.dim), dtype=np.float16)
        if isinstance(teacher, GAECacheTeacher):
            feat = teacher.episode(ep, length)
            for b in range(0, length, 64):
                grid, pool = to_grid(feat[b : b + 64].to(device))
                gbuf[b : b + 64] = grid.cpu().numpy().astype(np.float16)
                pbuf[b : b + 64] = pool.cpu().numpy().astype(np.float16)
            del feat
        else:
            frames = [G.decode_frames(G.video_path(args.dataset_dir, cam, ep), range(length)) for cam in G.CAMERAS]
            for b in range(0, length, args.batch):
                idx = list(range(b, min(b + args.batch, length)))
                feat = teacher([[frames[c][k] for c in range(len(G.CAMERAS))] for k in idx])
                grid, pool = to_grid(feat)
                gbuf[idx] = grid.cpu().numpy().astype(np.float16)
                pbuf[idx] = pool.cpu().numpy().astype(np.float16)
            del frames
        for path, buf in ((gpath, gbuf), (ppath, pbuf)):
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, buf)
            tmp.rename(path)
        done += 1
        print(f"episode {ep}: {length} steps, {done}/{len(mine)} done, {time.time() - t0:.0f}s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
