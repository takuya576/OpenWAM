#!/usr/bin/env python3
"""How much of the RGF loss is geometry? Loss of trivial students on a teacher cache (scripts/geometry/rgf_cache.py).

Uses the training code's relation and Huber functions (openwam/model/rgf_align.py, loaded by file path so any env
works). Samples 4 random 32-step windows per episode (seed 0) and, per window, the latent-frame offsets +0/+16/+32.

    const  every token / step gets the same vector, so every relation is 1
    mean   one constant relation per matrix, equal to the mean of the teacher's off-diagonal relations

Only RGF loss below the `mean` floor reflects relational structure beyond the overall similarity level.

    python scripts/geometry/rgf_trivial_floors.py assets/rgf_cache/hang_ring_vggt
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("rgf_align", ROOT / "openwam" / "model" / "rgf_align.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["rgf_align"] = R
_spec.loader.exec_module(R)


def main(cache):
    meta = json.load(open(f"{cache}/meta.json"))
    rng = np.random.default_rng(0)
    huber = lambda a, b: F.smooth_l1_loss(a, b, reduction="none", beta=1.0)  # noqa: E731
    vi = R.view_token_indices()
    out = {k: [] for k in ("intra", "cross", "temporal")}

    def add(term, r):
        out[term].append(
            (
                r.mean().item(),
                r.std().item(),
                huber(r, torch.ones_like(r)).mean().item(),
                huber(r, r.mean().expand_as(r)).mean().item(),
            )
        )

    for e, length in meta["lengths"].items():
        g = np.load(f"{cache}/episode_{int(e):06d}_grid.npy", mmap_mode="r")
        p = np.load(f"{cache}/episode_{int(e):06d}_pool.npy", mmap_mode="r")
        for s in rng.integers(0, length - 32, 4):
            for f in (s, s + 16, s + 32):
                x = torch.from_numpy(np.asarray(g[min(f, length - 1)], np.float32))
                for ix in vi:
                    add("intra", R._relation(x[ix])[~torch.eye(len(ix), dtype=torch.bool)])
                for a in range(len(vi)):
                    for b in range(a + 1, len(vi)):
                        add("cross", R._relation(x[vi[a]], x[vi[b]]).flatten())
            sc = torch.from_numpy(np.asarray(p[s : s + 32], np.float32)).reshape(32, -1)
            add("temporal", R._relation(sc)[~torch.eye(32, dtype=torch.bool)])
    print(f"{cache} (teacher {meta['teacher']}, D={meta['dim']}, {len(meta['lengths'])} episodes)")
    print("  term       relation mean  spread(std)  const floor  mean floor")
    for k, v in out.items():
        v = np.array(v).mean(0)
        print(f"  {k:9s}  {v[0]:13.3f}  {v[1]:11.3f}  {v[2]:11.4f}  {v[3]:10.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
