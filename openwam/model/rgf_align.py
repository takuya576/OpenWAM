"""Relational Geometry Forcing (RGF) baseline (``training.rgf``): a training-only auxiliary loss.

Re-implementation of "Relational Geometry Forcing for World Action Models" (arXiv 2610.29282,
Section 3.2 and Appendix A.2) for the OpenWAM dual-system joint self-attention model.
Procedure, settings and every deviation from the paper: ``docs/rgf_baseline.md``.

Three relational terms on the hidden states after ``layer`` blocks of BOTH experts:

* intra-frame   for every latent frame t and camera view v: the cosine relation matrix of the
                view's video tokens (projected by ``g_vid``) vs that of the teacher patches
                resampled to the same token grid; diagonal excluded.
* cross-view    for every view pair v < u at the same t: cross-relation matrices.
* action-temporal  the 32x32 relation matrix of the action tokens (projected by ``g_act``) vs the
                step-to-step relation of the scene vectors s_t = [pool(P_t^1); ...; pool(P_t^V)]
                of the observations at the same 32 steps; diagonal and padded steps excluded.

Each matrix pair is compared element-wise with a Huber (smooth-L1, beta=1) loss averaged over the
valid entries; the terms are averaged over views / view pairs, latent frames and samples, and

    L = L_vFM + L_aFM + weight * (w_intra * L_intra + w_cross * L_cross + w_temporal * L_temporal).

Pieces
------
* :class:`RGFCacheReader`  dataset side: teacher targets written by ``scripts/geometry/rgf_cache.py``
  (``episode_XXXXXX_grid.npy``  (T, 120, D) teacher patches resampled to the 12x10 canvas token grid,
  ``episode_XXXXXX_pool.npy``  (T, V, D) per-view spatial means of the full-resolution patches).
* :class:`RGFProjector`    model side: ``video`` (g_vid) and ``action`` (g_act) two-layer SiLU MLPs.
  Lives on the architecture as ``rgf_projector``; kept out of the deploy checkpoint and saved
  separately as ``rgf_projector_step_N.safetensors``.
* :func:`rgf_loss`         the three terms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

RGF_STATE_DICT_PREFIX = "rgf_projector."
RGF_PROJECTOR_FILE_PREFIX = "rgf_projector_step_"  # must NOT start with "checkpoint_step_"
RGF_PROJECTOR_FORMAT = "openwam.rgf_projector.v1"

# (row0, row1, col0, col1) of each camera view on the 12x10 token grid of the YAM canvas
# (384x320, 32 px per token), in the camera order of the cache: head, left wrist, right wrist.
YAM_VIEW_SLOTS: Tuple[Tuple[int, int, int, int], ...] = ((0, 8, 0, 10), (8, 12, 0, 5), (8, 12, 5, 10))
YAM_GRID: Tuple[int, int] = (12, 10)


def view_token_indices(slots=YAM_VIEW_SLOTS, grid=YAM_GRID) -> List[Tensor]:
    """Flat token indices (row-major on ``grid``) of every view slot."""
    gh, gw = grid
    out = []
    for r0, r1, c0, c1 in slots:
        if not (0 <= r0 < r1 <= gh and 0 <= c0 < c1 <= gw):
            raise ValueError(f"view slot {(r0, r1, c0, c1)} outside the {grid} token grid")
        rows = torch.arange(r0, r1)[:, None] * gw
        out.append((rows + torch.arange(c0, c1)[None, :]).reshape(-1))
    return out


def latent_frame_positions(num_video_frames: int, temporal_compression: int = 4) -> np.ndarray:
    """Position (in the sampled video clip) of the RGB frame that stands for each latent frame.

    Wan2.2's causal VAE encodes frame 0 alone and every later group of ``temporal_compression``
    frames into one latent frame; the LAST frame of each group is used (0, 4, 8 for 9 frames).
    """
    if (num_video_frames - 1) % temporal_compression:
        raise ValueError(f"{num_video_frames} video frames is not 1 + k*{temporal_compression}")
    return np.arange(0, num_video_frames, temporal_compression, dtype=np.int64)


@dataclass(frozen=True)
class RGFConfig:
    """``training.rgf`` options. See configs/train.yaml for the documented block."""

    layer: int = 15  # hidden states after this many blocks of each expert (1-based)
    weight: float = 0.5  # lambda_RGF
    w_intra: float = 1.0
    w_cross: float = 1.0
    w_temporal: float = 1.0
    proj_dim: int = 2048  # output width of g_vid and g_act (paper: 2048)
    hidden_dim: int = 2048  # hidden width of the two-layer heads
    huber_beta: float = 1.0

    @classmethod
    def from_training_cfg(cls, training_cfg) -> Optional["RGFConfig"]:
        node = training_cfg.get("rgf", None) if training_cfg is not None else None
        if node is None or not bool(node.get("enabled", False)):
            return None
        d = cls()
        cfg = cls(**{k: type(getattr(d, k))(node.get(k, getattr(d, k))) for k in d.__dataclass_fields__})
        if cfg.layer < 1:
            raise ValueError("training.rgf.layer counts blocks from 1")
        if cfg.weight <= 0 or min(cfg.w_intra, cfg.w_cross, cfg.w_temporal) < 0:
            raise ValueError("training.rgf: weight must be > 0 and the term weights >= 0")
        return cfg


class RGFCacheReader:
    """Random access to the per-episode teacher target files (memory-mapped, opened lazily per worker)."""

    def __init__(self, cache_dir: str):
        self._dir = Path(cache_dir)
        meta_path = self._dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"RGF cache: {meta_path} not found (run scripts/geometry/rgf_cache.py)")
        self.meta = json.loads(meta_path.read_text())
        if tuple(self.meta.get("grid", ())) != YAM_GRID:
            raise ValueError(f"RGF cache grid {self.meta.get('grid')} != canvas token grid {YAM_GRID}")
        self._maps: Dict[Tuple[int, str], np.ndarray] = {}

    def _open(self, episode: int, kind: str) -> np.ndarray:
        key = (episode, kind)
        mm = self._maps.get(key)
        if mm is None:
            path = self._dir / f"episode_{episode:06d}_{kind}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"RGF cache: {path} missing")
            mm = self._maps[key] = np.load(path, mmap_mode="r")
        return mm

    def _get(self, episode: int, frames: Sequence[int], kind: str) -> Tensor:
        mm = self._open(int(episode), kind)
        idx = np.clip(np.asarray(frames, dtype=np.int64), 0, mm.shape[0] - 1)
        return torch.from_numpy(np.ascontiguousarray(mm[idx]))

    def grid(self, episode: int, frames: Sequence[int]) -> Tensor:
        """``(len(frames), 120, D)`` float16 teacher patches on the canvas token grid; past-the-end frames repeat the last."""
        return self._get(episode, frames, "grid")

    def pool(self, episode: int, frames: Sequence[int]) -> Tensor:
        """``(len(frames), V, D)`` float16 per-view pooled teacher features; past-the-end frames repeat the last."""
        return self._get(episode, frames, "pool")


def _head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))


class RGFProjector(nn.Module):
    """g_vid and g_act: two-layer MLPs with SiLU (paper App. B.1: 3072 -> 2048 and 1024 -> 2048)."""

    def __init__(self, video_dim: int, action_dim: int, hidden_dim: int = 2048, proj_dim: int = 2048):
        super().__init__()
        self.video = _head(video_dim, hidden_dim, proj_dim)
        self.action = _head(action_dim, hidden_dim, proj_dim)


def _relation(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """Cosine relation matrix over the second-to-last axis: (..., N, D) [, (..., M, D)] -> (..., N, M).

    Always float32, also under an ambient autocast: the teacher's relations span a narrow band (mean intra-view cosine
    about 0.9), so a bf16 matmul's rounding (~4e-3) would be a sizeable part of the target signal.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        xn = F.normalize(x.float(), dim=-1)
        yn = xn if y is None else F.normalize(y.float(), dim=-1)
        return xn @ yn.transpose(-1, -2)


def _masked_mean(err: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    """Mean of ``err`` over the last two axes where ``mask`` is True -> (values, has_any)."""
    if mask is None:
        return err.mean(dim=(-1, -2)), torch.ones(err.shape[:-2], dtype=torch.bool, device=err.device)
    m = mask.to(err.dtype)
    n = m.sum(dim=(-1, -2))
    return (err * m).sum(dim=(-1, -2)) / n.clamp_min(1.0), n > 0


def _valid_mean(per: Tensor, valid: Optional[Tensor]) -> Tensor:
    """Mean of ``per`` over entries where ``valid`` (same shape) is True; 0 (with grad) if none."""
    if valid is None:
        return per.mean()
    w = valid.to(per.dtype)
    return (per * w).sum() / w.sum().clamp_min(1.0)


def rgf_loss(
    video_feat: Tensor,
    video_target: Tensor,
    video_valid: Optional[Tensor],
    action_feat: Tensor,
    scene: Tensor,
    action_valid: Optional[Tensor],
    view_index: Sequence[Tensor],
    beta: float = 1.0,
) -> Dict[str, Tensor]:
    """The three RGF terms.

    video_feat    (B, F, N, P)   projected video tokens of F latent frames on the N-token canvas grid
    video_target  (B, F, N, D)   teacher patches resampled to the same grid
    video_valid   (B, F) bool    False for latent frames whose RGB frame is past the episode end
    action_feat   (B, T, P)      projected action tokens
    scene         (B, T, V, D)   per-view pooled teacher features of the observation at each action step
    action_valid  (B, T) bool    False for padded action steps
    view_index    V tensors of flat token indices (one per camera view)
    """
    huber = lambda a, b: F.smooth_l1_loss(a, b, reduction="none", beta=beta)  # noqa: E731
    dev = video_feat.device
    view_index = [ix.to(dev) for ix in view_index]

    # intra-frame: per view, all token pairs except the diagonal
    intra = []
    for ix in view_index:
        xs, xg = video_feat[:, :, ix], video_target[:, :, ix]
        n = ix.numel()
        off = ~torch.eye(n, dtype=torch.bool, device=dev)
        per, _ = _masked_mean(huber(_relation(xs), _relation(xg)), off)  # (B, F)
        intra.append(per)
    l_intra = _valid_mean(torch.stack(intra, 0).mean(0), video_valid)

    # cross-view: every pair v < u, all entries
    cross = []
    for a in range(len(view_index)):
        for b in range(a + 1, len(view_index)):
            ia, ib = view_index[a], view_index[b]
            rs = _relation(video_feat[:, :, ia], video_feat[:, :, ib])
            rg = _relation(video_target[:, :, ia], video_target[:, :, ib])
            per, _ = _masked_mean(huber(rs, rg), None)
            cross.append(per)
    l_cross = _valid_mean(torch.stack(cross, 0).mean(0), video_valid) if cross else video_feat.sum() * 0

    # action-temporal: step-to-step relations, diagonal and padded steps excluded
    b_, t_ = action_feat.shape[:2]
    s = scene.reshape(b_, t_, -1)
    mask = ~torch.eye(t_, dtype=torch.bool, device=dev)[None].expand(b_, t_, t_)
    if action_valid is not None:
        av = action_valid.to(device=dev, dtype=torch.bool)
        mask = mask & av[:, :, None] & av[:, None, :]
    per, has = _masked_mean(huber(_relation(action_feat), _relation(s)), mask)  # (B,)
    l_temporal = _valid_mean(per, has)
    return {"intra": l_intra, "cross": l_cross, "temporal": l_temporal}


def exclude_rgf_from_state_dict(state_dict: dict) -> dict:
    """Drop the training-only projection heads so checkpoints deploy and warm-start like any other."""
    return {k: v for k, v in state_dict.items() if not k.startswith(RGF_STATE_DICT_PREFIX)}


def rgf_projector_path(output_dir: str, global_step: int) -> str:
    return str(Path(output_dir) / f"{RGF_PROJECTOR_FILE_PREFIX}{int(global_step)}.safetensors")


def save_rgf_projector(path: str, state_dict: dict, model, global_step: int) -> int:
    """Write the projection heads of a gathered architecture ``state_dict`` to their own file."""
    from safetensors.torch import save_file

    heads = {
        k[len(RGF_STATE_DICT_PREFIX) :]: v.detach().to("cpu").contiguous()
        for k, v in state_dict.items()
        if k.startswith(RGF_STATE_DICT_PREFIX)
    }
    if not heads:
        raise RuntimeError("rgf: no rgf_projector.* tensors in the state dict")
    cfg = getattr(model, "_rgf_cfg", None)
    meta = {
        "format": RGF_PROJECTOR_FORMAT,
        "global_step": str(int(global_step)),
        "rgf": json.dumps(cfg.__dict__ if cfg is not None else {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    save_file(heads, tmp, metadata=meta)
    Path(tmp).replace(path)
    return len(heads)
