"""Wan VAE / text-encoder IO with native-VAE ↔ external-encoder routing.

Free functions — ``vae`` / ``encoder`` / ``tokenizer`` / ``text_encoder`` passed
explicitly so the backbone class holds no IO logic. ``encoder is not None``
selects the external-encoder path; otherwise the native Wan VAE is used.
"""

from __future__ import annotations

import contextlib
import os
from collections import OrderedDict
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.preprocess import preprocess_video as _preprocess_video_native
from openwam.model.video_backbone.wan.preprocess import vae_output_to_video


def encode_text(prompts: list, *, tokenizer, text_encoder, device) -> Tuple[Tensor, Tensor]:
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        context[i, v:] = 0
    return context, seq_lens


def _module_device(module, fallback) -> torch.device:
    param = next(iter(module.parameters()), None) if hasattr(module, "parameters") else None
    return param.device if param is not None else torch.device(fallback)


@contextlib.contextmanager
def _cpu_encode_threads(device: torch.device):
    """Widen torch's intra-op pool for a CPU encode.

    torchrun pins every rank to ``OMP_NUM_THREADS=1``; a single-threaded
    umT5-XXL forward would take minutes. Each rank borrows its share of the
    cores for the (rare) cache miss and then restores the pool.
    """
    if device.type != "cpu":
        yield
        return
    previous = torch.get_num_threads()
    share = (os.cpu_count() or 1) // max(int(os.environ.get("LOCAL_WORLD_SIZE", "1")), 1)
    torch.set_num_threads(max(previous, min(32, share)))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class PromptEmbeddingCache:
    """Training-side memo of per-prompt text-encoder embeddings.

    The text encoder is frozen, so a prompt's embedding never changes, and a
    fine-tuning set typically has a handful of prompts. Serving them from this
    cache removes the per-step encoder forward (umT5-XXL over 512 padded tokens
    — about a quarter of a Wan2.2-5B training step) and lets the encoder live
    on CPU (``text_encoder_offload``), where only cache misses run.

    Each entry keeps just the valid tokens; :meth:`encode` rebuilds exactly the
    zero-padded ``(context, seq_lens)`` pair :func:`encode_text` returns.
    Entries are evicted least-recently-used beyond ``max_entries``.
    """

    def __init__(self, max_entries: int = 4096):
        self.max_entries = int(max_entries)
        self._store: "OrderedDict[str, Tensor]" = OrderedDict()
        self._padded_len: Optional[int] = None
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, prompt: str) -> bool:
        return prompt in self._store

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], *, tokenizer, text_encoder, device) -> Tuple[Tensor, Tensor]:
        device = torch.device(device)
        missing = [p for p in dict.fromkeys(prompts) if p not in self._store]
        if missing:
            # Encode where the encoder lives (CPU when offloaded), store on `device`.
            enc_device = _module_device(text_encoder, device)
            with _cpu_encode_threads(enc_device):
                context, seq_lens = encode_text(missing, tokenizer=tokenizer, text_encoder=text_encoder, device=enc_device)
            self._padded_len = int(context.shape[1])
            for prompt, emb, n in zip(missing, context, seq_lens.tolist()):
                self._store[prompt] = emb[: int(n)].to(device=device).clone()
        n_missed = sum(1 for p in prompts if p in missing)
        self.misses += n_missed
        self.hits += len(prompts) - n_missed

        entries = [self._store[p] for p in prompts]
        out = torch.zeros(
            (len(prompts), self._padded_len, entries[0].shape[-1]), dtype=entries[0].dtype, device=device
        )
        for i, (prompt, emb) in enumerate(zip(prompts, entries)):
            out[i, : emb.shape[0]] = emb
            self._store.move_to_end(prompt)
        seq_lens = torch.tensor([emb.shape[0] for emb in entries], dtype=torch.long, device=device)
        # Evict only after assembling, so a batch wider than max_entries still resolves.
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
        return out, seq_lens


def encode_text_for_inference(
    prompt, *, vace_cache, prompt_embed_cache, tokenizer, text_encoder, device
) -> Tuple[Tensor, Tensor]:
    """Deploy ``(context, seq_lens)``, reusing a prompt-keyed cached embed."""
    if vace_cache and vace_cache.get("populated") and vace_cache.get("prompt_key") == prompt:
        return vace_cache["context"], vace_cache["seq_lens"]
    if prompt_embed_cache is not None and prompt in prompt_embed_cache:
        return prompt_embed_cache[prompt]
    context, seq_lens = encode_text([prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=device)
    if prompt_embed_cache is not None:
        prompt_embed_cache[prompt] = (context, seq_lens)
    return context, seq_lens


def preprocess_video(frames, *, encoder=None, dtype, device) -> Tensor:
    if encoder is not None:
        return encoder.preprocess_video(frames)
    return _preprocess_video_native(frames, dtype=dtype, device=device)


def encode_video(video_tensor: Tensor, *, vae, encoder=None) -> Tensor:
    if encoder is not None:
        return encoder.batch_encode(video_tensor)
    return vae.batch_encode(video_tensor, device=video_tensor.device)


def encode_video_for_vace(
    pixels: Tensor,
    *,
    vae,
    encoder=None,
    device,
    tiled: bool,
    tile_size: tuple,
    tile_stride: tuple,
) -> Tensor:
    """Tiled-aware VAE encode for the VACE pixel→latent helper, returning
    ``(B, z_dim, T_lat, H_lat, W_lat)``. ``tiled=False`` (training) batches in
    one call; ``tiled=True`` (deploy) loops per-sample with bounded peak memory,
    mirroring the vendored unit so large-frame deploy does not OOM.
    """
    if not tiled:
        return encode_video(pixels, vae=vae, encoder=encoder).to(dtype=pixels.dtype, device=pixels.device)
    if encoder is not None:
        # No generic tiled-encode contract; fall back to batch_encode.
        # Unreachable today (VACE + external_encoder is fail-fast).
        return encoder.batch_encode(pixels).to(dtype=pixels.dtype, device=pixels.device)
    # Native Wan VAE: per-sample tiled encode (deploy is B=1).
    outs = []
    for i in range(pixels.shape[0]):
        lat = vae.encode(
            [pixels[i]],
            device=device,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        outs.append(lat)
    return torch.cat(outs, dim=0).to(dtype=pixels.dtype, device=pixels.device)


def decode_latents(latents: Tensor, *, vae, encoder=None, device, tiled: bool = True) -> Tensor:
    if encoder is not None:
        return encoder.decode(latents.to(device), tiled=tiled)
    return vae.decode(latents.to(device), device=device, tiled=tiled)


def latents_to_frames(video_tensor: Tensor, *, encoder=None) -> list:
    if encoder is not None:
        return encoder.to_frames(video_tensor)
    return vae_output_to_video(video_tensor)
