"""Tests for stateless trainer helpers (cross-rank reduce / debug-CSV writer).

Pure compute + IO — no GPU, no real Accelerator.
"""

import sys
import types

import pytest
import torch
from omegaconf import OmegaConf

from openwam.train.utils.training_utils import (
    build_constant_scheduler,
    build_cosine_scheduler,
    init_wandb,
    reduce_step_metrics,
    write_debug_loss_row,
)

# --- reduce_step_metrics (single-process fast path) ---


def test_reduce_step_metrics_single_process_passthrough():
    losses = {
        "total": torch.tensor(1.5),
        "video": torch.tensor(0.5),
        "action": 0.25,  # plain float exercises the _f() branch
    }
    out = reduce_step_metrics(None, losses, torch.tensor(2.0))
    assert out == {
        "loss_total": 1.5,
        "loss_video": 0.5,
        "loss_action": 0.25,
        "grad_norm": 2.0,
    }


def test_reduce_step_metrics_num_processes_one_uses_fast_path():
    class _SingleProc:
        num_processes = 1  # no gather() — fast path must not call it

    losses = {k: torch.tensor(0.0) for k in ("total", "video", "action")}
    losses["total"] = torch.tensor(1.0)
    out = reduce_step_metrics(_SingleProc(), losses, torch.tensor(0.0))
    assert out["loss_total"] == 1.0


# --- build_cosine_scheduler (num_processes scaling) ---


def _cosine_cfg(lr=1e-4, warmup_ratio=0.1, lr_min_ratio=0.01):
    training = types.SimpleNamespace(learning_rate=lr, warmup_ratio=warmup_ratio, lr_min_ratio=lr_min_ratio)
    return types.SimpleNamespace(training=training)


def _build(total_opt_steps, num_processes, lr=1e-4, warmup_ratio=0.1, lr_min_ratio=0.01):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    cfg = _cosine_cfg(lr=lr, warmup_ratio=warmup_ratio, lr_min_ratio=lr_min_ratio)
    return build_cosine_scheduler(opt, total_opt_steps=total_opt_steps, cfg=cfg, num_processes=num_processes)


def test_cosine_scheduler_scales_horizons_by_num_processes():
    # warmup_ratio=0.1 -> warmup=100, cosine=900; AcceleratedScheduler steps n times per opt step,
    # so inner horizons must be multiplied by n to land on the right opt step.
    sched = _build(total_opt_steps=1000, num_processes=8)
    warmup, cosine = sched._schedulers
    assert warmup.total_iters == 100 * 8
    assert cosine.T_max == 900 * 8
    assert sched._milestones == [100 * 8]
    assert cosine.eta_min == 1e-4 * 0.01


def test_cosine_scheduler_single_process_is_unscaled():
    sched = _build(total_opt_steps=1000, num_processes=1)
    warmup, cosine = sched._schedulers
    assert warmup.total_iters == 100
    assert cosine.T_max == 900
    assert sched._milestones == [100]


# --- build_constant_scheduler ---


def test_constant_scheduler_warms_up_then_holds():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    sched = build_constant_scheduler(opt, total_opt_steps=1000, cfg=_cosine_cfg(warmup_ratio=0.1), num_processes=1)
    lrs = []
    for _ in range(1000):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-4 / 100)
    assert lrs[50] < lrs[99] < 1e-4
    assert lrs[100:] == pytest.approx([1e-4] * 900)


def test_constant_scheduler_scales_warmup_by_num_processes():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    sched = build_constant_scheduler(opt, total_opt_steps=1000, cfg=_cosine_cfg(warmup_ratio=0.1), num_processes=8)
    assert sched.total_iters == 100 * 8


# --- init_wandb run naming ---


def _init_wandb_name(monkeypatch, *, training_run_name=None, wandb_run_name=None):
    calls = {}

    def fake_init(**kwargs):
        calls.update(kwargs)
        return types.SimpleNamespace(name=kwargs["name"])

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=fake_init))
    cfg = OmegaConf.create(
        {
            "training": {"run_name": training_run_name},
            "project": {"wandb": {"project": "openwam", "run_name": wandb_run_name, "entity": None}},
        }
    )
    init_wandb(cfg, "outputs/openwam_checkpoints/place_cube_right_n14/2026-10-27_23-10-00/")
    return calls["name"]


def test_init_wandb_names_run_after_its_checkpoint_dir(monkeypatch):
    name = _init_wandb_name(monkeypatch, training_run_name="place_cube_right_n14")
    assert name == "place_cube_right_n14_2026-10-27_23-10-00"
    assert _init_wandb_name(monkeypatch) == "2026-10-27_23-10-00"
    assert _init_wandb_name(monkeypatch, training_run_name="x", wandb_run_name="mine") == "mine"


# --- write_debug_loss_row ---


def test_write_debug_loss_row_header_then_append(tmp_path):
    labels = [("action", "loss_action")]
    metrics = {
        "loss_total": 1.0,
        "loss_video": 0.5,
        "loss_action": 0.25,
        "grad_norm": 2.0,
    }
    kw = dict(labels=labels, metrics=metrics, epoch=0, lr=1e-4, steps_per_sec=3.0)
    write_debug_loss_row(str(tmp_path), global_step=1, opt_step=0, **kw)
    write_debug_loss_row(str(tmp_path), global_step=2, opt_step=1, **kw)

    lines = (tmp_path / "debug_loss_history.csv").read_text().splitlines()
    assert lines[0] == ("step,opt_step,epoch,loss,loss_video,loss_action,grad_norm,lr,steps_per_sec")
    assert len(lines) == 3  # header written once, two data rows
    assert lines[1].startswith("1,0,0,")
    assert lines[2].startswith("2,1,0,")


def test_write_debug_loss_row_columns_follow_labels(tmp_path):
    labels = [("custom", "loss_custom")]
    metrics = {"loss_total": 1.0, "loss_video": 0.5, "loss_custom": 0.3, "grad_norm": 1.0}
    write_debug_loss_row(
        str(tmp_path),
        labels=labels,
        metrics=metrics,
        global_step=1,
        opt_step=0,
        epoch=0,
        lr=1e-4,
        steps_per_sec=1.0,
    )
    header = (tmp_path / "debug_loss_history.csv").read_text().splitlines()[0]
    assert header == ("step,opt_step,epoch,loss,loss_video,loss_custom,grad_norm,lr,steps_per_sec")
