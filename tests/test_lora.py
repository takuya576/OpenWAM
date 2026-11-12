"""LoRA fine-tuning: adapter math, trainability policy, merged checkpoints, prompt cache.

CPU-only. The toy architecture mirrors the real module tree (``video_backbone.dit``
blocks with q/k/v/o + ffn Sequential + norms + ``modulation``, ``head``,
``patch_embedding``, a frozen ``text_encoder`` / ``vae``, an ``action_backbone``
and a top-level ``proprio_encoder``) so the default ``training.lora`` selection
is exercised against the same dotted names it targets in production.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from openwam.model.lora import (
    DEFAULT_FULL_TRAIN,
    LoRAConfig,
    LoRALinear,
    apply_lora,
    inject_lora,
    is_lora_param_name,
    lora_modules,
    lora_scalings,
    merge_lora_,
    merge_lora_state_dict,
    merge_lora_weight,
)

# ---------------------------------------------------------------------------
# Toy model with the production module layout
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class _Attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.norm_q = _RMSNorm(dim)

    def forward(self, x):
        attn = torch.softmax(self.norm_q(self.q(x)) @ self.k(x).transpose(-1, -2), dim=-1)
        return self.o(attn @ self.v(x))


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = _Attn(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) * 0.1)

    def forward(self, x):
        shift, scale = self.modulation.chunk(2, dim=1)
        x = x + self.self_attn(self.norm3(x) * (1 + scale) + shift)
        return x + self.ffn(x)


class _Head(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.head = nn.Linear(dim, out)
        self.modulation = nn.Parameter(torch.zeros(1, 2, dim))

    def forward(self, x):
        return self.head(x + self.modulation[:, 0])


class _DiT(nn.Module):
    def __init__(self, dim=16, in_dim=4):
        super().__init__()
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=1)
        self.text_embedding = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([_Block(dim), _Block(dim)])
        self.head = _Head(dim, in_dim)

    def forward(self, x, text):
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2) + self.text_embedding(text)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens)


class _VideoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = _DiT()
        self.text_encoder = nn.Linear(8, 8)
        self.vae = nn.Linear(4, 4)


class _ActionBackbone(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.action_encoder = nn.Linear(6, dim)
        self.blocks = nn.ModuleList([_Block(dim)])
        self.action_decoder = nn.Linear(dim, 6)

    def forward(self, a, ctx):
        h = self.action_encoder(a) + ctx
        for block in self.blocks:
            h = block(h)
        return self.action_decoder(h)


class _Arch(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_backbone = _VideoBackbone()
        self.action_backbone = _ActionBackbone()
        self.proprio_encoder = nn.Linear(6, 8)

    def forward(self, x, text, actions, proprio):
        text = self.video_backbone.text_encoder(text) + self.proprio_encoder(proprio)
        video = self.video_backbone.dit(x, text)
        ctx = video.mean(dim=1, keepdim=True).repeat(1, 1, 4)
        return video, self.action_backbone(actions, ctx)

    def get_trainable_modules(self, freeze_list=()):
        return {n: m for n, m in self.named_children() if any(p.requires_grad for p in m.parameters())}


FROZEN = ["video_backbone.text_encoder", "video_backbone.vae"]


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(2, 4, 1, 2, 2, generator=g),
        torch.randn(2, 1, 8, generator=g),
        torch.randn(2, 3, 6, generator=g),
        torch.randn(2, 1, 6, generator=g),
    )


def _freeze(model):
    for name in FROZEN:
        model.get_submodule(name).requires_grad_(False)


def _lora_arch(seed=0, **cfg_kwargs):
    # Rank 4 on 16-wide toy layers (the production default 128 targets 3072-wide ones).
    cfg_kwargs.setdefault("rank", 4)
    cfg_kwargs.setdefault("alpha", 4)
    torch.manual_seed(seed)
    model = _Arch()
    _freeze(model)
    report = apply_lora(model, LoRAConfig(**cfg_kwargs), frozen=FROZEN)
    return model, report


@torch.no_grad()
def _randomize_adapters(model, scale=0.2, seed=1):
    g = torch.Generator().manual_seed(seed)
    for module in lora_modules(model).values():
        module.lora_B.copy_(torch.randn(module.lora_B.shape, generator=g) * scale)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------


def test_lora_linear_is_exact_noop_at_init_and_adopts_base_params():
    base = nn.Linear(12, 7)
    lora = LoRALinear(base, rank=4, alpha=8)
    x = torch.randn(3, 12)
    assert torch.equal(lora(x), base(x))
    assert lora.weight is base.weight and lora.bias is base.bias
    assert isinstance(lora, nn.Linear)
    assert lora.scaling == 2.0
    assert set(lora.state_dict()) == {"weight", "bias", "lora_A", "lora_B"}


def test_lora_linear_without_bias_matches_nn_linear():
    base = nn.Linear(5, 3, bias=False)
    lora = LoRALinear(base, rank=2, alpha=2)
    assert lora.bias is None and "bias" not in lora.state_dict()
    x = torch.randn(4, 5)
    assert torch.equal(lora(x), base(x))


def test_lora_linear_forward_equals_merged_weight():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(10, 6), rank=3, alpha=6)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(5, 10)
    expected = nn.functional.linear(x, lora.merged_weight(), lora.bias)
    torch.testing.assert_close(lora(x), expected, rtol=1e-5, atol=1e-5)


def test_merge_lora_weight_never_mutates_an_fp32_source():
    # .to(float32) on an fp32 tensor is a no-op alias; the merge must still copy.
    weight = torch.randn(6, 4)
    original = weight.clone()
    merged = merge_lora_weight(weight, torch.randn(2, 4), torch.randn(6, 2), 0.5)
    assert torch.equal(weight, original)
    assert not torch.equal(merged, original)


def test_merge_lora_weight_rounds_once_from_fp32():
    weight = torch.randn(8, 8, dtype=torch.bfloat16)
    a, b = torch.randn(2, 8, dtype=torch.bfloat16), torch.randn(8, 2, dtype=torch.bfloat16)
    merged = merge_lora_weight(weight, a, b, 0.25)
    exact = (weight.float() + 0.25 * b.float() @ a.float()).to(torch.bfloat16)
    assert merged.dtype == torch.bfloat16 and torch.equal(merged, exact)


def test_lora_dropout_is_inactive_in_eval():
    lora = LoRALinear(nn.Linear(6, 6), rank=2, alpha=2, dropout=0.5)
    with torch.no_grad():
        lora.lora_B.normal_()
    lora.eval()
    x = torch.randn(3, 6)
    assert torch.equal(lora(x), lora(x))


# ---------------------------------------------------------------------------
# Injection / trainability policy
# ---------------------------------------------------------------------------


def test_inject_lora_respects_targets_exclude_and_skip_prefixes():
    model = _Arch()
    adapted = inject_lora(
        model,
        targets=["video_backbone.dit"],
        exclude=["*.self_attn.v"],
        skip_prefixes=["video_backbone.dit.head"],
        rank=2,
        alpha=2,
    )
    assert "video_backbone.dit.blocks.0.self_attn.q" in adapted
    assert "video_backbone.dit.blocks.1.ffn.2" in adapted  # nn.Sequential children are reachable
    assert "video_backbone.dit.text_embedding.0" in adapted
    assert not any(name.endswith("self_attn.v") for name in adapted)
    assert "video_backbone.dit.head.head" not in adapted
    assert not any(name.startswith(("action_backbone", "video_backbone.text_encoder")) for name in adapted)
    # Idempotent: an already-adapted layer is never wrapped twice.
    assert inject_lora(model, targets=["video_backbone.dit"], rank=2, alpha=2) == [
        "video_backbone.dit.blocks.0.self_attn.v",
        "video_backbone.dit.blocks.1.self_attn.v",
        "video_backbone.dit.head.head",
    ]


def test_apply_lora_default_partition():
    model, report = _lora_arch()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    lora = {n for n in trainable if is_lora_param_name(n)}
    assert lora and all(n.startswith("video_backbone.dit.") for n in lora)
    # Pretrained DiT matrices are frozen; their adapters train.
    assert "video_backbone.dit.blocks.0.self_attn.q.weight" not in trainable
    assert "video_backbone.dit.blocks.0.ffn.0.weight" not in trainable
    # full_train: action expert, proprio encoder, DiT head + patch embedding.
    assert {n for n, _ in model.action_backbone.named_parameters(prefix="action_backbone")} <= trainable
    assert {"proprio_encoder.weight", "proprio_encoder.bias"} <= trainable
    assert {"video_backbone.dit.head.head.weight", "video_backbone.dit.patch_embedding.weight"} <= trainable
    # bias / norm / modulation inside the DiT.
    assert {
        "video_backbone.dit.blocks.0.self_attn.q.bias",
        "video_backbone.dit.blocks.0.self_attn.norm_q.weight",
        "video_backbone.dit.blocks.0.norm3.weight",
        "video_backbone.dit.blocks.0.modulation",
    } <= trainable
    # The freeze-list modules stay frozen.
    assert not any(n.startswith(("video_backbone.text_encoder", "video_backbone.vae")) for n in trainable)
    assert report.trainable_params == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert report.trainable_params + report.frozen_params == sum(p.numel() for p in model.parameters())


def test_apply_lora_without_bias_and_norm_trains_only_adapters_and_full_train():
    model, report = _lora_arch(train_bias_and_norm=False)
    assert report.bias_norm_params == 0
    assert not model.video_backbone.dit.blocks[0].modulation.requires_grad
    assert not model.video_backbone.dit.blocks[0].self_attn.q.bias.requires_grad


def test_apply_lora_rejects_training_a_frozen_module_and_unknown_targets():
    model = _Arch()
    with pytest.raises(ValueError, match="freeze list"):
        apply_lora(model, LoRAConfig(targets=("video_backbone",)), frozen=FROZEN)
    with pytest.raises(ValueError, match="do not exist"):
        apply_lora(_Arch(), LoRAConfig(targets=("video_backbone.nope",)), frozen=FROZEN)


def test_apply_lora_skips_absent_full_train_entries():
    model = _Arch()
    del model.proprio_encoder
    report = apply_lora(model, LoRAConfig(), frozen=FROZEN)
    assert report.skipped_full_train == ["proprio_encoder"]
    assert "skipped" in report.summary()


def test_gradients_reach_adapters_and_full_train_but_not_frozen_weights():
    model, _ = _lora_arch()
    _randomize_adapters(model)  # B != 0 so dL/dA is non-zero too
    video, actions = model(*_inputs())
    (video.square().mean() + actions.square().mean()).backward()
    q = model.video_backbone.dit.blocks[0].self_attn.q
    assert q.weight.grad is None
    assert q.lora_A.grad is not None and q.lora_A.grad.abs().sum() > 0
    assert q.lora_B.grad is not None and q.lora_B.grad.abs().sum() > 0
    assert model.action_backbone.action_encoder.weight.grad.abs().sum() > 0
    assert model.video_backbone.text_encoder.weight.grad is None


def test_one_optimizer_step_moves_only_trainable_params():
    model, _ = _lora_arch()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    video, actions = model(*_inputs())
    (video.square().mean() + actions.square().mean()).backward()
    opt.step()
    for name, param in model.named_parameters():
        changed = not torch.equal(param, before[name])
        if not param.requires_grad:
            assert not changed, f"frozen param {name} moved"
    assert not torch.equal(model.video_backbone.dit.blocks[0].self_attn.q.lora_B, before["video_backbone.dit.blocks.0.self_attn.q.lora_B"])


# ---------------------------------------------------------------------------
# Merged checkpoints
# ---------------------------------------------------------------------------


def test_merged_state_dict_has_base_keys_and_reproduces_lora_outputs():
    model, _ = _lora_arch()
    _randomize_adapters(model)
    model.eval()
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    merged = merge_lora_state_dict(sd, lora_scalings(model))

    plain = _Arch()
    assert set(merged) == set(plain.state_dict())
    plain.load_state_dict(merged, strict=True)
    plain.eval()
    with torch.no_grad():
        for got, want in zip(plain(*_inputs(3)), model(*_inputs(3))):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # The source state dict is untouched (live weights must not change while saving).
    assert all(torch.equal(sd[k], v) for k, v in model.state_dict().items())


def test_merge_lora_state_dict_bf16_within_rounding():
    model, _ = _lora_arch()
    _randomize_adapters(model, scale=0.05)
    model.to(torch.bfloat16).eval()
    merged = merge_lora_state_dict(model.state_dict(), lora_scalings(model))
    plain = _Arch().to(torch.bfloat16)
    plain.load_state_dict(merged, strict=True)
    plain.eval()
    with torch.no_grad():
        inputs = [t.to(torch.bfloat16) for t in _inputs(4)]
        for got, want in zip(plain(*inputs), model(*inputs)):
            torch.testing.assert_close(got.float(), want.float(), rtol=0.05, atol=0.05)


def test_merge_lora_state_dict_rejects_unmerged_leftovers():
    model, _ = _lora_arch()
    scalings = lora_scalings(model)
    scalings.pop(next(iter(scalings)))
    with pytest.raises(KeyError, match="without a scaling"):
        merge_lora_state_dict(model.state_dict(), scalings)


def test_merge_lora_in_place_restores_plain_structure():
    model, _ = _lora_arch()
    _randomize_adapters(model)
    model.eval()
    with torch.no_grad():
        want = model(*_inputs(5))
    merged = merge_lora_(model)
    assert merged and not lora_modules(model)
    assert set(model.state_dict()) == set(_Arch().state_dict())
    with torch.no_grad():
        for got, ref in zip(model(*_inputs(5)), want):
            torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_save_weights_writes_a_merged_checkpoint(tmp_path):
    from safetensors.torch import load_file, save_file

    from openwam.train.utils.checkpointing import save_weights

    model, _ = _lora_arch()
    _randomize_adapters(model)
    model.eval()
    model.save_checkpoint = lambda path, state_dict: save_file(state_dict, path)

    accelerator = SimpleNamespace(
        is_main_process=True,
        device=torch.device("cpu"),
        unwrap_model=lambda m: m,
        get_state_dict=lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()},
    )
    save_weights(accelerator, model, str(tmp_path), 7, final=True)

    saved = load_file(str(tmp_path / "checkpoint_step_7.safetensors"))
    assert not any(is_lora_param_name(k) for k in saved)
    plain = _Arch()
    plain.load_state_dict(saved, strict=True)
    plain.eval()
    with torch.no_grad():
        for got, want in zip(plain(*_inputs(6)), model(*_inputs(6))):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # The live model keeps training unmerged.
    assert lora_modules(model)


# ---------------------------------------------------------------------------
# Config + optimizer groups
# ---------------------------------------------------------------------------


def test_lora_config_from_training_cfg():
    assert LoRAConfig.from_training_cfg(OmegaConf.create({"lr": 1e-4})) is None
    assert LoRAConfig.from_training_cfg(OmegaConf.create({"lora": {"enabled": False}})) is None
    cfg = LoRAConfig.from_training_cfg(
        OmegaConf.create(
            {"lora": {"enabled": True, "rank": 64, "alpha": 32, "lr": "5e-4", "targets": "video_backbone.dit"}}
        )
    )
    assert cfg.rank == 64 and cfg.alpha == 32.0 and cfg.lr == 5e-4
    assert cfg.targets == ("video_backbone.dit",)
    assert cfg.full_train == DEFAULT_FULL_TRAIN


def test_train_yaml_lora_block_parses_and_is_disabled():
    from pathlib import Path

    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "train.yaml")
    assert LoRAConfig.from_training_cfg(cfg.training) is None
    cfg.training.lora.enabled = True
    parsed = LoRAConfig.from_training_cfg(cfg.training)
    assert parsed == LoRAConfig()


def test_optimizer_groups_split_adapters_only_with_lora_lr():
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    model, report = _lora_arch()
    trainer = SimpleNamespace(architecture=model)

    flat = build_trainable_parameters(trainer)
    assert isinstance(flat, list) and sum(p.numel() for p in flat) == report.trainable_params

    groups = build_trainable_parameters(trainer, lora_lr=1e-3)
    by_lr = {g.get("lr"): g["params"] for g in groups}
    assert set(by_lr) == {None, 1e-3}
    assert sum(p.numel() for p in by_lr[1e-3]) == report.lora_params
    assert sum(p.numel() for g in groups for p in g["params"]) == report.trainable_params


# ---------------------------------------------------------------------------
# Real architecture class
# ---------------------------------------------------------------------------


def test_apply_lora_on_a_real_architecture_round_trips_through_save_load(tmp_path):
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    report = apply_lora(
        arch,
        LoRAConfig(targets=("video_backbone._dit",), full_train=("action_backbone",)),
        frozen=["video_backbone._text_encoder", "video_backbone._vae"],
    )
    assert report.adapted == ["video_backbone._dit"]
    _randomize_adapters(arch)
    merged = merge_lora_state_dict(arch.state_dict(), lora_scalings(arch))
    path = str(tmp_path / "ckpt.safetensors")
    arch.save_checkpoint(path, state_dict=merged)

    fresh = _make_tiny_arch()
    fresh.load_checkpoint(path)  # strict: identical key set
    x = torch.randn(2, 64)
    torch.testing.assert_close(fresh.video_backbone._dit(x), arch.video_backbone._dit(x), rtol=1e-5, atol=1e-5)
