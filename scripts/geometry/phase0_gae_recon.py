#!/usr/bin/env python3
"""Phase 0A: does GAE's 64-channel latent keep the geometry of YAM robot views?

For held-out episodes of one task, each camera frame is encoded two ways and
decoded back to depth through the frozen DA3 DPT head:

* single : the frame alone (a one-view DA3 set) — what a policy can compute live.
* clip   : the frame as view 0 of a 17-frame clip at stride 4 (the codec was
           trained on clips of 17-81 views).

Each GAE-decoded depth is compared with the depth the same DPT head reads from
the *uncompressed* DA3 features of the same set (the "teacher"):

* abs_rel / delta1        : median-scaled depth agreement over the whole image
* abs_rel_edge_band       : the same, only within ``--band`` px of teacher depth edges
* edge precision/recall/F1: depth-edge maps (|grad log depth| > ``--tau``) matched
                            within ``--tol`` px
* rgb_psnr                : GAE RGB head vs the input frame (sanity of the codec)
* set_effect_*            : teacher(single) vs teacher(clip) — how much the DA3 set
                            itself changes the geometry of frame 0

Writes ``records.csv`` (one row per frame x camera x mode), ``summary.json``
(means by camera type) and ``panels/*.png`` to ``--out-dir``.

Run in the ``gae`` conda env:
    CUDA_VISIBLE_DEVICES=4 python scripts/geometry/phase0_gae_recon.py \\
        --dataset-dir assets/datasets/place_cube_right --episodes 56:60 \\
        --out-dir outputs/geometry_phase0/place_cube_right/recon
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gae_tools as G  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--episodes", required=True, help="half-open range, e.g. 56:60 (held-out episodes)")
    p.add_argument("--quantiles", default="0.1,0.45,0.8", help="clip start positions within each episode")
    p.add_argument("--clip-views", type=int, default=17)
    p.add_argument("--clip-stride", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.03, help="depth-edge threshold on |grad log depth| per pixel")
    p.add_argument("--tol", type=int, default=2, help="edge match radius in pixels (at 378x504)")
    p.add_argument("--band", type=int, default=3, help="edge-band half-width for abs_rel_edge_band")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ── visualization ───────────────────────────────────────────────────────────


def _depth_rgb(depth: torch.Tensor, lo: float, hi: float) -> np.ndarray:
    import matplotlib.cm as cm

    inv = 1.0 / depth.clamp_min(1e-4).cpu().numpy()  # near = bright
    norm = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    return (cm.viridis(norm)[..., :3] * 255).astype(np.uint8)


def _edge_overlay(image: torch.Tensor, ref: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    """Teacher edges green, GAE edges magenta, both white, on a darkened gray image."""
    gray = (image.mean(0).cpu().numpy() * 0.45 * 255).astype(np.uint8)
    out = np.stack([gray] * 3, axis=-1)
    r, p = ref.cpu().numpy(), pred.cpu().numpy()
    out[r & ~p] = (40, 220, 40)
    out[p & ~r] = (230, 40, 230)
    out[r & p] = (255, 255, 255)
    return out


def _tile(arr: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((G.DA3_W // 2, G.DA3_H // 2), Image.Resampling.BILINEAR))


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


# ── main ────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve()
    device = torch.device(args.device)
    model = G.load_gae(device)

    lengths = G.episode_lengths(dataset_dir)
    quantiles = [float(q) for q in args.quantiles.split(",")]
    span = (args.clip_views - 1) * args.clip_stride
    records = []
    t_start = time.time()

    for ep in G.parse_range(args.episodes):
        n = lengths[ep]
        starts = sorted({int(round(q * (n - 1 - span))) for q in quantiles})
        needed = sorted({s + k * args.clip_stride for s in starts for k in range(args.clip_views)})
        frames = {cam: G.decode_frames(G.video_path(dataset_dir, cam, ep), needed) for cam in G.CAMERAS}

        for start in starts:
            clip_idx = [start + k * args.clip_stride for k in range(args.clip_views)]
            # single: the three cameras as three independent one-view sets
            x_single = G.to_da3_input([frames[cam][start] for cam in G.CAMERAS]).to(device)
            feats, mu = G.encode(model, x_single.unsqueeze(1))
            raw_s = G.raw_depth(model, feats, G.DA3_H, G.DA3_W)
            gae_s = G.gae_depth(model, mu, G.DA3_H, G.DA3_W)
            rgb_s = G.gae_rgb(model, mu, num_views=1)

            panel_rows = []
            for ci, cam in enumerate(G.CAMERAS):
                short = G.CAMERA_SHORT[cam]
                x_clip = G.to_da3_input([frames[cam][i] for i in clip_idx]).to(device)
                feats_c, mu_c = G.encode(model, x_clip.unsqueeze(0))
                raw_c = G.raw_depth(model, feats_c, G.DA3_H, G.DA3_W)
                gae_c = G.gae_depth(model, mu_c, G.DA3_H, G.DA3_W)
                rgb_c = G.gae_rgb(model, mu_c, num_views=args.clip_views)

                def compare(pred, ref):
                    ref_e = G.log_depth_gradient(ref.unsqueeze(0))[0] > args.tau
                    pred_e = G.log_depth_gradient(pred.unsqueeze(0))[0] > args.tau
                    band = G.dilate(ref_e.unsqueeze(0), args.band)[0]
                    m = G.depth_agreement(pred, ref, band)
                    m.update({f"edge_{k}": v for k, v in G.edge_f1(pred_e, ref_e, args.tol).items()})
                    return m, ref_e, pred_e

                base = {"episode": ep, "frame": start, "camera": short}
                m_single, ref_e, pred_e = compare(gae_s[ci], raw_s[ci])
                m_single["rgb_psnr"] = G.psnr(rgb_s[ci], x_single[ci])
                records.append({**base, "mode": "single", **m_single})

                m_clip0, _, _ = compare(gae_c[0], raw_c[0])
                m_clip0["rgb_psnr"] = G.psnr(rgb_c[0], x_clip[0])
                records.append({**base, "mode": "clip_view0", **m_clip0})

                per_view = [compare(gae_c[v], raw_c[v])[0] for v in range(args.clip_views)]
                m_all = {k: float(np.nanmean([pv[k] for pv in per_view])) for k in per_view[0]}
                m_all["rgb_psnr"] = float(np.mean([G.psnr(rgb_c[v], x_clip[v]) for v in range(args.clip_views)]))
                records.append({**base, "mode": "clip_all_views", **m_all})

                m_set, _, _ = compare(raw_c[0], raw_s[ci])
                records.append({**base, "mode": "set_effect_teacher", **m_set})

                inv = 1.0 / raw_s[ci].clamp_min(1e-4)
                lo, hi = float(torch.quantile(inv.flatten()[::7], 0.02)), float(torch.quantile(inv.flatten()[::7], 0.98))
                panel_rows.append(
                    np.concatenate(
                        [
                            _tile(_to_uint8(x_single[ci])),
                            _tile(_to_uint8(rgb_s[ci])),
                            _tile(_depth_rgb(raw_s[ci], lo, hi)),
                            _tile(_depth_rgb(gae_s[ci], lo, hi)),
                            _tile(_depth_rgb(gae_c[0], lo, hi)),
                            _tile(_edge_overlay(x_single[ci], ref_e, pred_e)),
                        ],
                        axis=1,
                    )
                )
            Image.fromarray(np.concatenate(panel_rows, axis=0)).save(
                out_dir / "panels" / f"ep{ep:03d}_f{start:04d}.png"
            )
        print(f"[ep {ep}] {len(starts)} clips done ({time.time() - t_start:.0f}s)", flush=True)

    keys = list(dict.fromkeys(k for r in records for k in r))
    with open(out_dir / "records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(records)

    groups = defaultdict(list)
    for r in records:
        cam_type = "head" if r["camera"] == "head" else "wrist"
        groups[(r["mode"], cam_type)].append(r)
    metric_keys = [k for k in keys if k not in ("episode", "frame", "camera", "mode")]
    summary = {
        "dataset_dir": str(dataset_dir),
        "episodes": args.episodes,
        "n_frames_per_camera_type": {f"{m}/{c}": len(v) for (m, c), v in groups.items()},
        "settings": {k: getattr(args, k) for k in ("tau", "tol", "band", "clip_views", "clip_stride", "quantiles")},
        "panel_columns": ["input", "GAE RGB (single)", "DA3 depth (single)", "GAE depth (single)",
                          "GAE depth (clip, view 0)", "edges: DA3 green / GAE magenta / both white"],
        "mean": {
            f"{m}/{c}": {
                k: float(np.nanmean([r[k] for r in v if k in r])) for k in metric_keys if any(k in r for r in v)
            }
            for (m, c), v in sorted(groups.items())
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["mean"], indent=1))


if __name__ == "__main__":
    main()
