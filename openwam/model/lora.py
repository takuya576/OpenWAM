"""Low-rank adapters (LoRA) for parameter-efficient OpenWAM fine-tuning.

``training.lora`` (configs/train.yaml) trains a low-rank update ``s * B @ A`` on
the ``nn.Linear`` layers of the target modules (the 5B video DiT by default)
while their pretrained weights stay frozen, and trains the ``full_train``
modules (the action expert, the proprio encoder and the DiT's tiny patch
embedding / output head by default) in full. The DiT's biases, normalization
weights and AdaLN modulation tensors are optionally trained in full as well.

Design points:

* :class:`LoRALinear` subclasses ``nn.Linear`` and adopts the wrapped layer's
  ``weight`` / ``bias`` Parameter objects, so the base state-dict keys and every
  ``isinstance(m, nn.Linear)`` check are unchanged; only ``lora_A`` / ``lora_B``
  are added. ``B`` starts at zero, so an adapted model is bit-identical to the
  pretrained one until the first optimizer step.
* Checkpoints are written MERGED (:func:`merge_lora_state_dict`): each adapter
  is folded into its base weight in fp32 and rounded once to the weight dtype,
  and the adapter keys are dropped. A LoRA checkpoint therefore has exactly the
  key set of a full fine-tune, so deploy, open-loop eval and a later
  ``finetune_ckpt_path`` load it unchanged.
* Resume (``resume_ckpt_path``) restores the UNMERGED adapters from accelerate's
  full state, which works because the trainer injects the adapters before
  ``accelerator.prepare`` on every launch.
"""

from __future__ import annotations

import fnmatch
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

#: Parameter names :class:`LoRALinear` adds next to ``weight`` / ``bias``.
LORA_PARAM_NAMES = ("lora_A", "lora_B")


class LoRALinear(nn.Linear):
    """``nn.Linear`` plus a trainable low-rank update.

    ``y = x W^T + b + scaling * dropout(x) A^T B^T`` with ``A: (rank, in)``,
    ``B: (out, rank)`` and ``scaling = alpha / rank``. ``A`` is initialized like
    ``nn.Linear`` (Kaiming-uniform) and ``B`` with zeros, so the adapter is an
    exact no-op at initialization.

    The wrapped layer's ``weight`` / ``bias`` Parameter objects are adopted, not
    copied: no extra memory, and the state-dict keys stay ``<name>.weight`` /
    ``<name>.bias``.
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0):
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        # Skip nn.Linear.__init__: it would allocate (and initialize) a fresh weight.
        nn.Module.__init__(self)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        # register_parameter keeps a bias-less layer's `bias` a registered None, as in nn.Linear.
        self.register_parameter("bias", base.bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, **factory))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank, **factory))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Carry the base layer's train/eval flag over to the replacement.
        self.train(base.training)
        self.reset_lora_parameters()

    @torch.no_grad()
    def reset_lora_parameters(self) -> None:
        """Kaiming-uniform ``A`` (as ``nn.Linear`` / PEFT) and zero ``B``.

        ``A`` is drawn in fp32 and then cast, so a bf16 model gets the same
        initialization distribution as an fp32 one.
        """
        if self.lora_A.device.type == "meta":
            return
        a = torch.empty(self.lora_A.shape, dtype=torch.float32, device=self.lora_A.device)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A.copy_(a)
        self.lora_B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + update * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """``W + scaling * B @ A`` computed in fp32 and rounded once to ``W``'s dtype."""
        return merge_lora_weight(self.weight, self.lora_A, self.lora_B, self.scaling)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, rank={self.rank}, alpha={self.alpha:g}"


@torch.no_grad()
def merge_lora_weight(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    *,
    compute_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Fold one adapter into its base weight: ``W + scaling * B @ A``.

    The update is computed and added in fp32 and the sum is rounded ONCE to
    ``weight.dtype``, so merging a bf16 model loses no more precision than
    storing the fp32 master weights of a full fine-tune in bf16. The result
    lives on ``weight``'s device; ``compute_device`` only picks where the fp32
    math runs (e.g. the GPU while saving a CPU state dict).
    """
    device = compute_device or weight.device
    # copy=True: for an fp32 weight already on `device`, .to() would return the
    # weight itself and the in-place addmm_ would corrupt the source tensor.
    merged = weight.to(device=device, dtype=torch.float32, copy=True)
    lora_b = lora_b.to(device=device, dtype=torch.float32)
    lora_a = lora_a.to(device=device, dtype=torch.float32)
    merged.addmm_(lora_b, lora_a, alpha=scaling)
    return merged.to(device=weight.device, dtype=weight.dtype)


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------


def _under(name: str, prefix: str) -> bool:
    """True when the dotted ``name`` is ``prefix`` itself or lives below it."""
    return name == prefix or name.startswith(prefix + ".")


def _under_any(name: str, prefixes: Iterable[str]) -> bool:
    return any(_under(name, p) for p in prefixes)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    """fnmatch-glob match on the dotted name (``*`` also crosses dots)."""
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


# ---------------------------------------------------------------------------
# Injection / inspection
# ---------------------------------------------------------------------------


def inject_lora(
    model: nn.Module,
    *,
    targets: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    exclude: Sequence[str] = (),
    skip_prefixes: Sequence[str] = (),
) -> list[str]:
    """Replace every plain ``nn.Linear`` under a ``targets`` prefix with a :class:`LoRALinear`.

    Layers whose dotted name matches an ``exclude`` glob or lives under a
    ``skip_prefixes`` entry are left alone. Only exact ``nn.Linear`` instances
    are adapted: subclasses may override ``forward`` (and an existing
    :class:`LoRALinear` must not be wrapped twice). Returns the adapted names.
    """
    adapted: list[str] = []
    for name, module in list(model.named_modules()):
        if type(module) is not nn.Linear:
            continue
        if not _under_any(name, targets) or _matches_any(name, exclude) or _under_any(name, skip_prefixes):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        adapted.append(name)
    return adapted


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    """``{dotted name: module}`` for every :class:`LoRALinear` in ``model``."""
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_scalings(model: nn.Module) -> dict[str, float]:
    """``{dotted name: scaling}`` of every adapter — what :func:`merge_lora_state_dict` needs."""
    return {name: m.scaling for name, m in lora_modules(model).items()}


def is_lora_param_name(name: str) -> bool:
    """True for the ``lora_A`` / ``lora_B`` parameters a :class:`LoRALinear` adds."""
    return name.rsplit(".", 1)[-1] in LORA_PARAM_NAMES


@torch.no_grad()
def merge_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    scalings: Mapping[str, float],
    *,
    compute_device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    """Return a copy of ``state_dict`` with every adapter folded into its base weight.

    ``scalings`` maps each adapted layer's dotted name to its ``alpha / rank``
    (:func:`lora_scalings`). The returned dict has the key set of the
    un-adapted model: ``<name>.weight`` is replaced by the merged weight and
    ``<name>.lora_A`` / ``<name>.lora_B`` are dropped. Tensors of ``state_dict``
    are never modified in place.
    """
    merged = dict(state_dict)
    for name, scaling in scalings.items():
        try:
            lora_a = merged.pop(f"{name}.lora_A")
            lora_b = merged.pop(f"{name}.lora_B")
            weight = merged[f"{name}.weight"]
        except KeyError as exc:
            raise KeyError(f"state_dict lacks the weight or adapter of LoRA layer {name!r}: {exc}") from exc
        merged[f"{name}.weight"] = merge_lora_weight(weight, lora_a, lora_b, scaling, compute_device=compute_device)
    leftover = [k for k in merged if is_lora_param_name(k)]
    if leftover:
        raise KeyError(f"adapter tensors without a scaling entry (not merged): {leftover[:5]}")
    return merged


@torch.no_grad()
def merge_lora_(model: nn.Module) -> list[str]:
    """Fold every adapter into its base layer IN PLACE and swap back plain ``nn.Linear``s.

    Leaves a model whose structure and state-dict keys are those of the
    un-adapted architecture (e.g. for in-process export / evaluation).
    Returns the merged layer names.
    """
    merged: list[str] = []
    for name, module in list(lora_modules(model).items()):
        plain = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device="meta",
        )
        plain.weight = nn.Parameter(module.merged_weight(), requires_grad=module.weight.requires_grad)
        plain.bias = module.bias
        plain.train(module.training)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, plain)
        merged.append(name)
    return merged


# ---------------------------------------------------------------------------
# Config-driven setup used by the trainer
# ---------------------------------------------------------------------------

#: Default full-train set: the action expert learns the new embodiment's action
#: space at full capacity; the DiT's patch embedding and output head are tiny
#: (~1.2M params) input/output layers whose rank cap would only cost quality.
DEFAULT_FULL_TRAIN = (
    "action_backbone",
    "proprio_encoder",
    "video_backbone.dit.head",
    "video_backbone.dit.patch_embedding",
)


@dataclass(frozen=True)
class LoRAConfig:
    """``training.lora`` options. See configs/train.yaml for the documented block."""

    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0
    lr: Optional[float] = None
    targets: tuple[str, ...] = ("video_backbone.dit",)
    exclude: tuple[str, ...] = ()
    full_train: tuple[str, ...] = DEFAULT_FULL_TRAIN
    train_bias_and_norm: bool = True

    @classmethod
    def from_training_cfg(cls, training_cfg) -> Optional["LoRAConfig"]:
        """Build from ``cfg.training``; ``None`` when ``training.lora.enabled`` is false/absent."""
        node = getattr(training_cfg, "lora", None) if training_cfg is not None else None
        if node is None or not bool(_get(node, "enabled", False)):
            return None
        defaults = cls()
        lr = _get(node, "lr", None)
        return cls(
            rank=int(_get(node, "rank", defaults.rank)),
            alpha=float(_get(node, "alpha", defaults.alpha)),
            dropout=float(_get(node, "dropout", defaults.dropout)),
            lr=None if lr is None else float(lr),
            targets=_str_tuple(_get(node, "targets", defaults.targets)),
            exclude=_str_tuple(_get(node, "exclude", defaults.exclude)),
            full_train=_str_tuple(_get(node, "full_train", defaults.full_train)),
            train_bias_and_norm=bool(_get(node, "train_bias_and_norm", defaults.train_bias_and_norm)),
        )


def _get(node, key, default=None):
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


def _str_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(v) for v in value)


@dataclass
class LoRAReport:
    """What :func:`apply_lora` adapted and made trainable (for logging / tests)."""

    adapted: list[str] = field(default_factory=list)
    skipped_full_train: list[str] = field(default_factory=list)
    lora_params: int = 0
    full_train_params: int = 0
    bias_norm_params: int = 0
    frozen_params: int = 0

    @property
    def trainable_params(self) -> int:
        return self.lora_params + self.full_train_params + self.bias_norm_params

    def summary(self) -> str:
        total = self.trainable_params + self.frozen_params
        pct = 100.0 * self.trainable_params / max(total, 1)
        text = (
            f"[lora] {len(self.adapted)} Linear layers adapted | trainable {self.trainable_params / 1e6:,.1f}M "
            f"of {total / 1e6:,.1f}M ({pct:.2f}%): adapters {self.lora_params / 1e6:,.1f}M, "
            f"full-train {self.full_train_params / 1e6:,.1f}M, bias/norm {self.bias_norm_params / 1e6:,.2f}M"
        )
        if self.skipped_full_train:
            text += f" | full_train entries absent on this model (skipped): {self.skipped_full_train}"
        return text


def _is_bias_or_norm(param_name: str, owner: nn.Module) -> bool:
    """Biases, normalization-layer weights and AdaLN ``modulation`` tensors."""
    leaf = param_name.rsplit(".", 1)[-1]
    return leaf == "bias" or leaf == "modulation" or "Norm" in type(owner).__name__


def apply_lora(model: nn.Module, cfg: LoRAConfig, *, frozen: Sequence[str] = ()) -> LoRAReport:
    """Inject adapters and set ``requires_grad`` for LoRA training.

    After this call exactly these parameters are trainable:

    * every adapter (``lora_A`` / ``lora_B``) on the ``nn.Linear`` layers under
      ``cfg.targets`` (minus ``cfg.exclude`` and the ``full_train`` layers);
    * every parameter under a ``cfg.full_train`` prefix;
    * with ``cfg.train_bias_and_norm``, the biases, norm weights and
      ``modulation`` tensors under ``cfg.targets``.

    Everything else is frozen (``requires_grad=False`` only: gradients must
    still flow THROUGH the frozen DiT to its adapters and to the action expert
    it attends with, so no ``no_grad`` wrapper is installed). ``frozen`` is the
    model yaml's ``freeze:`` list; those modules run under ``no_grad`` and so
    cannot be trained — overlapping them raises.
    """
    named_modules = dict(model.named_modules())
    missing_targets = [p for p in cfg.targets if p not in named_modules]
    if missing_targets:
        raise ValueError(
            f"training.lora.targets: {missing_targets} do not exist on {type(model).__name__}. "
            f"Top-level children: {[n for n, _ in model.named_children()]}"
        )
    # full_train entries absent on this architecture (e.g. proprio_encoder with
    # use_proprioception=false) are skipped and reported, not fatal.
    full_train = tuple(p for p in cfg.full_train if p in named_modules)
    for prefix in (*cfg.targets, *full_train):
        clash = [f for f in frozen if f in named_modules and (_under(prefix, f) or _under(f, prefix))]
        if clash:
            raise ValueError(
                f"training.lora: {prefix!r} overlaps the frozen (no_grad) modules {clash} from the "
                "model's freeze list; those cannot be trained."
            )

    adapted = inject_lora(
        model,
        targets=cfg.targets,
        rank=cfg.rank,
        alpha=cfg.alpha,
        dropout=cfg.dropout,
        exclude=cfg.exclude,
        skip_prefixes=full_train,
    )
    if not adapted:
        raise ValueError(f"training.lora: no nn.Linear layer found under targets={list(cfg.targets)}")

    report = LoRAReport(adapted=adapted, skipped_full_train=[p for p in cfg.full_train if p not in named_modules])
    owners = {name: module for name, module in model.named_modules()}
    for name, param in model.named_parameters():
        owner_name, _, leaf = name.rpartition(".")
        owner = owners.get(owner_name, model)
        if isinstance(owner, LoRALinear) and leaf in LORA_PARAM_NAMES:
            param.requires_grad_(True)
            report.lora_params += param.numel()
        elif _under_any(owner_name, full_train):
            param.requires_grad_(True)
            report.full_train_params += param.numel()
        elif cfg.train_bias_and_norm and _under_any(owner_name, cfg.targets) and _is_bias_or_norm(name, owner):
            param.requires_grad_(True)
            report.bias_norm_params += param.numel()
        else:
            param.requires_grad_(False)
            report.frozen_params += param.numel()
    return report


__all__ = [
    "DEFAULT_FULL_TRAIN",
    "LORA_PARAM_NAMES",
    "LoRAConfig",
    "LoRALinear",
    "LoRAReport",
    "apply_lora",
    "inject_lora",
    "is_lora_param_name",
    "lora_modules",
    "lora_scalings",
    "merge_lora_",
    "merge_lora_state_dict",
    "merge_lora_weight",
]
