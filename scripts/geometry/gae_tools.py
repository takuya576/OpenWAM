"""GAE / Depth Anything 3 helpers for OpenWAM geometry experiments.

Runs in the ``gae`` conda env (torch 2.5.1, depth-anything-3), not in ``openwam``.
Everything here reads YAM LeRobot v2.1 episodes directly and returns numpy /
torch arrays; nothing imports openwam, so the two environments never mix.

Conventions
-----------
* Shapes are H x W unless noted.
* DA3 / GAE run on each camera frame resized from 480x640 to 378x504
  (27x36 patches of 14 px, same 4:3 aspect as the recording).
* "Canvas" is OpenWAM's 384x320 three-camera L-shape
  (openwam/dataloader/transforms/multiview.py): head on top (256x320), left
  wrist bottom-left (128x160), right wrist bottom-right (128x160). Each camera
  is stretched into its slot, so per-camera maps are resampled into the slot
  the same way.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GAE_ROOT = Path(os.environ.get("GAE_ROOT", "/home/takuya/GAE-GeometricAutoEncoder"))

CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
CAMERA_SHORT = {CAMERAS[0]: "head", CAMERAS[1]: "left_wrist", CAMERAS[2]: "right_wrist"}
CANVAS_H, CANVAS_W = 384, 320
# camera -> (top, left, height, width) in canvas pixels (top_height_ratio 2/3).
SLOTS = {
    CAMERAS[0]: (0, 0, 256, 320),
    CAMERAS[1]: (256, 0, 128, 160),
    CAMERAS[2]: (256, 160, 128, 160),
}
DA3_H, DA3_W = 378, 504
WINDOW_FRAMES = 33  # yam.yaml num_frames: a window needs frames [f, f + 32]


# ── dataset access ──────────────────────────────────────────────────────────


def parse_range(spec: str) -> List[int]:
    """'56:66' -> [56, ..., 65] (half-open, like yam.yaml splits)."""
    start, end = (int(x) for x in spec.split(":"))
    return list(range(start, end))


def episode_lengths(dataset_dir: str | Path) -> Dict[int, int]:
    out = {}
    with open(Path(dataset_dir) / "meta" / "episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["episode_index"])] = int(row["length"])
    return out


def video_path(dataset_dir: str | Path, camera: str, episode: int, chunks_size: int = 1000) -> Path:
    chunk = episode // chunks_size
    return Path(dataset_dir) / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{episode:06d}.mp4"


def decode_frames(path: str | Path, indices: Iterable[int]) -> Dict[int, np.ndarray]:
    """Decode the requested frame indices (frame position in the file) as HxWx3 uint8.

    Sequential decode from frame 0: episodes are single short mp4s, and this
    indexes frames exactly as openwam's PyAV reader does (by position).
    """
    import av

    wanted = set(int(i) for i in indices)
    last = max(wanted)
    out: Dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"{path}: frames {sorted(missing)[:5]} not decoded")
    return out


def to_da3_input(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """List of HxWx3 uint8 -> (N, 3, 378, 504) float in [0, 1] (LANCZOS, as GAE's demo)."""
    arr = [
        np.asarray(Image.fromarray(f).resize((DA3_W, DA3_H), Image.Resampling.LANCZOS), dtype=np.float32)
        for f in frames
    ]
    return torch.from_numpy(np.stack(arr) / 255.0).permute(0, 3, 1, 2).contiguous()


# ── GAE / DA3 ───────────────────────────────────────────────────────────────


def load_gae(device: torch.device):
    """Released GAE-64 codec + frozen DA3-Giant (no flow model).

    GAE's configs resolve weights and DA3 statistics relative to its repo
    root, so construction runs from there.
    """
    for p in (str(GAE_ROOT), str(GAE_ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    cwd = os.getcwd()
    os.chdir(GAE_ROOT)
    try:
        from gae import GAE

        model = GAE.from_configs(
            str(GAE_ROOT / "configs" / "gae_64.yaml"),
            str(GAE_ROOT / "ckpts" / "gae_64.pt"),
            latent_stats=str(GAE_ROOT / "ckpts" / "latent_stats_gae_64.pt"),
            device=device,
        )
    finally:
        os.chdir(cwd)
    return model.eval()


@torch.inference_mode()
def encode(model, images: torch.Tensor) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """images (B, V, 3, H, W) in [0, 1] -> (raw DA3 features with CLS, posterior mean mu).

    Views inside one batch element form one DA3 set (cross-view attention);
    single-frame encoding is V=1. Mirrors ``GAE.encode`` (bf16 autocast).
    Returns features {level: (B*V, N+1, C)} and mu (B*V, 64, H/14, W/14).
    """
    b, v = images.shape[:2]
    h, w = images.shape[-2:]
    mean = model.backbone.encoder_mean.to(images).view(1, 1, 3, 1, 1)
    std = model.backbone.encoder_std.to(images).view(1, 1, 3, 1, 1)
    x = (images - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats_cls = model.backbone.encode(x, mode="all")
        feats = {k: f[:, 1:, :] for k, f in feats_cls.items()}
        xn = model.codec.normalize_levels(feats, image_size=(h, w))
        mu, _ = model.codec.encode(xn)
    return feats_cls, mu.float()


def _depth_to_bv(depth: torch.Tensor, bv: int, h: int, w: int) -> torch.Tensor:
    if depth.numel() != bv * h * w:
        raise RuntimeError(f"unexpected DPT depth shape {tuple(depth.shape)} for {bv} views of {h}x{w}")
    return depth.reshape(bv, h, w).float()


@torch.inference_mode()
def raw_depth(model, feats_cls: Dict[int, torch.Tensor], h: int, w: int) -> torch.Tensor:
    """Depth read by the frozen DA3 DPT head from the *uncompressed* DA3 features: (BV, H, W)."""
    from utils.dpt_helpers import _format_feats_for_dpt

    feats = {k: f.float() for k, f in feats_cls.items()}
    bv = next(iter(feats.values())).shape[0]
    dpt_in = _format_feats_for_dpt(feats, model.backbone.backbone_norm)
    with torch.autocast("cuda", enabled=False):
        out = model.backbone.rae_cl_decoder(dpt_in, h, w, patch_start_idx=0)
    return _depth_to_bv(out["depth"], bv, h, w)


@torch.inference_mode()
def da3_depth(model, images: torch.Tensor) -> torch.Tensor:
    """Teacher depth of independent single frames: images (N, 3, H, W) in [0, 1] -> (N, H, W).

    DA3 encoder + DPT head only (no GAE codec); each frame is its own one-view set.
    """
    h, w = images.shape[-2:]
    mean = model.backbone.encoder_mean.to(images).view(1, 3, 1, 1)
    std = model.backbone.encoder_std.to(images).view(1, 3, 1, 1)
    x = ((images - mean) / std).unsqueeze(1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats_cls = model.backbone.encode(x, mode="all")
    return raw_depth(model, feats_cls, h, w)


def canvas_slots(canvas: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a 384x320 canvas (H, W, 3) into its three camera images."""
    out = {}
    for camera, (top, left, h, w) in SLOTS.items():
        out[CAMERA_SHORT[camera]] = canvas[top : top + h, left : left + w]
    return out


@torch.inference_mode()
def gae_depth(model, mu: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Depth read by the same frozen DPT head from GAE-decoded features: (BV, H, W)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.decode_geometry(mu, h, w)
    return _depth_to_bv(out["depth"], mu.shape[0], h, w)


@torch.inference_mode()
def gae_rgb(model, mu: torch.Tensor, num_views: int) -> torch.Tensor:
    """GAE RGB head: (BV, 3, H, W) in [0, 1]. Batch must be contiguous (B, V) groups."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb = model.decode_rgb(mu, num_views=num_views)
    return rgb.float()


# ── geometry maps ───────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0


def log_depth_gradient(depth: torch.Tensor) -> torch.Tensor:
    """|grad log depth| per pixel (N, H, W): a scale-invariant depth-discontinuity strength.

    A 10% depth step spread over ~3 px gives ~0.03; a table plane seen from the
    head camera stays around 0.002-0.005.
    """
    ld = torch.log(depth.clamp_min(1e-4)).unsqueeze(1)
    ld = F.pad(ld, (1, 1, 1, 1), mode="replicate")
    kx = _SOBEL_X.to(ld).view(1, 1, 3, 3)
    ky = _SOBEL_X.t().contiguous().to(ld).view(1, 1, 3, 3)
    gx = F.conv2d(ld, kx)
    gy = F.conv2d(ld, ky)
    return torch.sqrt(gx * gx + gy * gy)[:, 0]


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation with a (2r+1)^2 square, mask (N, H, W) bool."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask.float().unsqueeze(1), k, stride=1, padding=radius)[:, 0] > 0.5


def to_canvas(per_camera: Dict[str, torch.Tensor], cell: int, reduce: str) -> torch.Tensor:
    """Place per-camera maps (N, C, H, W) into the canvas at ``cell``-px resolution.

    Each camera map is pooled into its slot's cell grid (``reduce`` = 'max' or
    'avg'), which also applies the slot's stretch. Returns (N, C, 384/cell, 320/cell).
    """
    any_map = next(iter(per_camera.values()))
    n, c = any_map.shape[:2]
    out = any_map.new_zeros(n, c, CANVAS_H // cell, CANVAS_W // cell)
    for camera, (top, left, h, w) in SLOTS.items():
        m = per_camera[camera]
        size = (h // cell, w // cell)
        pooled = F.adaptive_max_pool2d(m, size) if reduce == "max" else F.adaptive_avg_pool2d(m, size)
        out[:, :, top // cell : (top + h) // cell, left // cell : (left + w) // cell] = pooled
    return out


def slot_masks(cell: int) -> Dict[str, np.ndarray]:
    """Boolean (384/cell, 320/cell) masks of each camera slot."""
    masks = {}
    for camera, (top, left, h, w) in SLOTS.items():
        m = np.zeros((CANVAS_H // cell, CANVAS_W // cell), dtype=bool)
        m[top // cell : (top + h) // cell, left // cell : (left + w) // cell] = True
        masks[CAMERA_SHORT[camera]] = m
    return masks


# ── metrics ─────────────────────────────────────────────────────────────────


def depth_agreement(pred: torch.Tensor, ref: torch.Tensor, band: torch.Tensor | None = None) -> Dict[str, float]:
    """Median-scaled AbsRel and delta<1.25 of ``pred`` against ``ref`` (both (H, W), > 0).

    ``band`` optionally restricts AbsRel to a pixel subset (e.g. near depth edges).
    ``log_scale`` is log(median ref / median pred): the global scale drift removed.
    """
    pred = pred.clamp_min(1e-4)
    ref = ref.clamp_min(1e-4)
    s = ref.median() / pred.median()
    p = pred * s
    rel = (p - ref).abs() / ref
    ratio = torch.maximum(p / ref, ref / p)
    out = {
        "abs_rel": float(rel.mean()),
        "delta1": float((ratio < 1.25).float().mean()),
        "log_scale": float(torch.log(s)),
    }
    if band is not None and band.any():
        out["abs_rel_edge_band"] = float(rel[band].mean())
        out["abs_rel_off_band"] = float(rel[~band].mean()) if (~band).any() else float("nan")
    return out


def edge_f1(pred_edges: torch.Tensor, ref_edges: torch.Tensor, tol: int) -> Dict[str, float]:
    """Boundary precision / recall / F1 with a ``tol``-pixel match radius ((H, W) bool)."""
    pe, re_ = pred_edges.unsqueeze(0), ref_edges.unsqueeze(0)
    n_pred = float(pe.sum())
    n_ref = float(re_.sum())
    prec = float((pe & dilate(re_, tol)).sum()) / n_pred if n_pred else float("nan")
    rec = float((re_ & dilate(pe, tol)).sum()) / n_ref if n_ref else float("nan")
    if not n_pred and not n_ref:
        f1 = float("nan")  # neither map has an edge: nothing to score
    elif not n_pred or not n_ref:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "ref_edge_frac": n_ref / ref_edges.numel()}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))
