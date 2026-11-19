"""GAE alignment loss (``training.gae_align``): a training-only auxiliary loss.

The hidden states of the future-frame video tokens after one DiT block are mapped by a
small projector to GAE geometry latents of the ground-truth future frames, and compared
with them.  Procedure and picture: ``docs/gae_alignment.md``.

Pieces
------
* :class:`GAECacheReader`  dataset side — reads the targets written by
  ``scripts/geometry/gae_cache.py`` (one ``episode_XXXXXX.npy`` per episode,
  ``(T, cameras, 64, 27, 36)`` float16).
* :class:`GAEProjector`    model side — token hidden state -> a patch of GAE tokens for
  each video frame the token stands for.  Lives on the architecture as
  ``gae_projector``.  Kept out of the deploy checkpoint; saved for later analysis as a
  separate ``gae_projector_step_N.safetensors`` next to each ``checkpoint_step_N.safetensors``
  (:func:`save_projector` / :func:`load_projector`).  The deploy loader only reads
  ``checkpoint_step_*.safetensors``, so it never loads the projector.
* :func:`gae_align_loss`   cosine / MSE between prediction and target.

Shapes (YAM canvas 384x320, 9 video frames, Wan2.2 VAE + patch 2 -> 3 latent frames of
12x10 tokens):  future tokens ``(B, 2, 12, 10, D)``  ->  prediction
``(B, 8, 3, 64, 32, 40)`` = 8 future video frames x 3 cameras, each camera on a 32x40 grid
(head slot 8x10 tokens x 4x4, wrist slots 4x5 tokens x 8x8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

GAE_STATE_DICT_PREFIX = "gae_projector."
PROJECTOR_FILE_PREFIX = "gae_projector_step_"  # must NOT start with "checkpoint_step_" (deploy / pruning globs)
PROJECTOR_FORMAT = "openwam.gae_projector.v1"
LOSS_MODES = ("cosine", "mse", "cosine_mse")

# (row0, row1, col0, col1, sub) on the 12x10 token grid of the YAM canvas: the slot of each
# camera, in the camera order of the cache, and how many GAE tokens per side one video
# token predicts.  rows x sub and cols x sub must be equal for all slots (here 32 x 40).
YAM_SLOTS: Tuple[Tuple[int, int, int, int, int], ...] = (
    (0, 8, 0, 10, 4),  # head         (cam_high)
    (8, 12, 0, 5, 8),  # left wrist   (cam_left_wrist)
    (8, 12, 5, 10, 8),  # right wrist  (cam_right_wrist)
)


@dataclass(frozen=True)
class GAEAlignConfig:
    """``training.gae_align`` options. See configs/train.yaml for the documented block."""

    layer: int = 15  # hidden states after this many DiT blocks (1-based)
    weight: float = 0.5
    loss: str = "cosine"
    hidden_dim: int = 2048
    channels: int = 64

    @classmethod
    def from_training_cfg(cls, training_cfg) -> Optional["GAEAlignConfig"]:
        node = training_cfg.get("gae_align", None) if training_cfg is not None else None
        if node is None or not bool(node.get("enabled", False)):
            return None
        d = cls()
        cfg = cls(
            layer=int(node.get("layer", d.layer)),
            weight=float(node.get("weight", d.weight)),
            loss=str(node.get("loss", d.loss)),
            hidden_dim=int(node.get("hidden_dim", d.hidden_dim)),
            channels=int(node.get("channels", d.channels)),
        )
        if cfg.loss not in LOSS_MODES:
            raise ValueError(f"training.gae_align.loss must be one of {LOSS_MODES}, got {cfg.loss!r}")
        if cfg.layer < 1:
            raise ValueError("training.gae_align.layer counts DiT blocks from 1")
        return cfg


class GAECacheReader:
    """Random access to the per-episode GAE target files (memory-mapped, opened lazily per worker)."""

    def __init__(self, cache_dir: str):
        self._dir = Path(cache_dir)
        meta_path = self._dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"GAE cache: {meta_path} not found (run scripts/geometry/gae_cache.py)")
        self.meta = json.loads(meta_path.read_text())
        self._maps: Dict[int, np.ndarray] = {}

    def _open(self, episode: int) -> np.ndarray:
        mm = self._maps.get(episode)
        if mm is None:
            path = self._dir / f"episode_{episode:06d}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"GAE cache: {path} missing")
            mm = self._maps[episode] = np.load(path, mmap_mode="r")
        return mm

    def get(self, episode: int, frames: Sequence[int]) -> Tensor:
        """``(len(frames), cameras, 64, 27, 36)`` float16; frames past the episode end repeat the last frame."""
        mm = self._open(int(episode))
        idx = np.clip(np.asarray(frames, dtype=np.int64), 0, mm.shape[0] - 1)
        return torch.from_numpy(np.ascontiguousarray(mm[idx]))


class GAEProjector(nn.Module):
    """Video-token hidden state -> GAE latent patches, one per video frame the token covers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        channels: int = 64,
        frames_per_latent: int = 4,
        slots: Sequence[Tuple[int, int, int, int, int]] = YAM_SLOTS,
        grid: Tuple[int, int] = (12, 10),
    ):
        super().__init__()
        sizes = {((r1 - r0) * s, (c1 - c0) * s) for r0, r1, c0, c1, s in slots}
        if len(sizes) != 1:
            raise ValueError(f"GAEProjector: every slot must give the same target grid, got {sorted(sizes)}")
        self.slots = tuple(tuple(int(v) for v in s) for s in slots)
        self.grid = (int(grid[0]), int(grid[1]))
        self.out_size = next(iter(sizes))
        self.channels = int(channels)
        self.frames_per_latent = int(frames_per_latent)
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.frame_emb = nn.Parameter(torch.zeros(self.frames_per_latent, hidden_dim))
        nn.init.normal_(self.frame_emb, std=0.02)
        self.mid = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.ModuleDict(
            {str(s): nn.Linear(hidden_dim, s * s * self.channels) for s in sorted({s[4] for s in self.slots})}
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """``tokens`` (B, F, H, W, D) future latent frames -> (B, F * frames_per_latent, n_slots, C, h, w)."""
        b, f, hh, ww, _ = tokens.shape
        if (hh, ww) != self.grid:
            raise ValueError(f"GAEProjector: token grid {hh}x{ww}, expected {self.grid[0]}x{self.grid[1]}")
        k, c = self.frames_per_latent, self.channels
        x = F.silu(self.inp(tokens))  # (B, F, H, W, hid)
        x = x.unsqueeze(2) + self.frame_emb.view(1, 1, k, 1, 1, -1)  # (B, F, K, H, W, hid)
        x = F.silu(self.mid(x))
        outs = []
        for r0, r1, c0, c1, s in self.slots:
            y = self.out[str(s)](x[:, :, :, r0:r1, c0:c1])  # (B, F, K, h, w, s*s*C)
            h, w = r1 - r0, c1 - c0
            y = y.reshape(b, f, k, h, w, s, s, c).permute(0, 1, 2, 7, 3, 5, 4, 6)  # (B, F, K, C, h, s, w, s)
            outs.append(y.reshape(b, f * k, c, h * s, w * s))
        return torch.stack(outs, dim=2)


def gae_align_loss(pred: Tensor, target: Tensor, valid: Optional[Tensor], mode: str = "cosine") -> Tensor:
    """Alignment loss.

    pred    (B, T, S, C, h, w)    projector output (any float dtype; compared in float32)
    target  (B, T, S, C, h0, w0)  cached GAE latents; each camera grid is resized to (h, w)
    valid   (B, T) bool or None   False for frames past the end of the episode

    Every camera contributes the same weight: the per-location loss is averaged inside each
    camera first, then over cameras, frames and samples.
    """
    b, t, s, c, h, w = pred.shape
    if target.shape[:4] != (b, t, s, c):
        raise ValueError(f"gae_align_loss: prediction {tuple(pred.shape)} vs target {tuple(target.shape)}")
    pred = pred.float()
    tgt = target.reshape(b * t * s, c, *target.shape[-2:]).float()
    if tgt.shape[-2:] != (h, w):
        tgt = F.interpolate(tgt, size=(h, w), mode="bilinear", align_corners=False)
    tgt = tgt.reshape(b, t, s, c, h, w)
    per = pred.new_zeros(b, t, s, h, w)
    if mode in ("cosine", "cosine_mse"):
        per = per + (1.0 - F.cosine_similarity(pred, tgt, dim=3, eps=1e-6))
    if mode in ("mse", "cosine_mse"):
        per = per + (pred - tgt).pow(2).mean(dim=3)
    per = per.mean(dim=(3, 4))  # (B, T, S): mean over the locations of one camera
    if valid is None:
        return per.mean()
    wgt = valid.to(per.dtype)[:, :, None].expand_as(per)
    return (per * wgt).sum() / wgt.sum().clamp_min(1.0)


def exclude_gae_from_state_dict(state_dict: dict) -> dict:
    """Drop the training-only projector so checkpoints deploy and warm-start like any other."""
    return {k: v for k, v in state_dict.items() if not k.startswith(GAE_STATE_DICT_PREFIX)}


def projector_path(output_dir: str, global_step: int) -> str:
    return str(Path(output_dir) / f"{PROJECTOR_FILE_PREFIX}{int(global_step)}.safetensors")


def save_projector(path: str, state_dict: dict, model, global_step: int) -> int:
    """Write the projector weights of a gathered architecture ``state_dict`` to their own file.

    The file is self-describing: its safetensors metadata holds everything needed to
    rebuild the module (:func:`load_projector`) and the alignment settings it was trained
    with.  Returns the number of tensors written.
    """
    from safetensors.torch import save_file

    proj = model.gae_projector
    cfg = getattr(model, "_gae_align_cfg", None)
    tensors = {
        k[len(GAE_STATE_DICT_PREFIX) :]: v.detach().contiguous().cpu()
        for k, v in state_dict.items()
        if k.startswith(GAE_STATE_DICT_PREFIX)
    }
    expected = set(proj.state_dict().keys())
    if set(tensors) != expected:
        raise RuntimeError(
            f"save_projector: state dict has projector keys {sorted(tensors)[:4]}..., expected {sorted(expected)[:4]}..."
        )
    meta = {
        "format": PROJECTOR_FORMAT,
        "global_step": str(int(global_step)),
        "in_dim": str(int(proj.inp.in_features)),
        "hidden_dim": str(int(proj.inp.out_features)),
        "channels": str(proj.channels),
        "frames_per_latent": str(proj.frames_per_latent),
        "slots": json.dumps([list(s) for s in proj.slots]),
        "grid": json.dumps(list(proj.grid)),
        "align": json.dumps({"layer": cfg.layer, "weight": cfg.weight, "loss": cfg.loss} if cfg is not None else {}),
        "note": "training-only projector of training.gae_align; not part of the deploy checkpoint",
    }
    tmp = f"{path}.tmp"
    save_file(tensors, tmp, metadata=meta)
    Path(tmp).replace(path)
    return len(tensors)


def load_projector(path: str, map_location: str = "cpu", dtype: Optional[torch.dtype] = None):
    """Rebuild a saved projector for analysis. Returns ``(GAEProjector, metadata dict)``."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, "pt") as f:
        meta = dict(f.metadata() or {})
    if meta.get("format") != PROJECTOR_FORMAT:
        raise ValueError(f"load_projector: {path} is not a {PROJECTOR_FORMAT} file (format={meta.get('format')!r})")
    proj = GAEProjector(
        in_dim=int(meta["in_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        channels=int(meta["channels"]),
        frames_per_latent=int(meta["frames_per_latent"]),
        slots=[tuple(s) for s in json.loads(meta["slots"])],
        grid=tuple(json.loads(meta["grid"])),
    )
    proj.load_state_dict(load_file(path, device=map_location), strict=True)
    proj = proj.to(map_location)
    if dtype is not None:
        proj = proj.to(dtype)
    info = {
        "global_step": int(meta["global_step"]),
        "align": json.loads(meta.get("align", "{}")),
        "slots": json.loads(meta["slots"]),
        "grid": json.loads(meta["grid"]),
    }
    return proj.eval(), info
