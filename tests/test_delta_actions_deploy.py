"""Delta-action dims are bound to the checkpoint, so deploy overrides cannot change them.

A checkpoint trained with ``dataloader.delta_action_dims`` predicts those dims
relative to the chunk-start state, and the engine adds the request state back.
The dims must come from the checkpoint's own ``config.yaml``: a deploy override
that dropped them would send relative offsets to the robot as absolute joint
targets, and one that added them would add the state twice. Servers are built
through ``build_server_from_config`` with the model build and weight loading
stubbed, so the real load -> bind -> merge -> engine path runs on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import openwam.model
from openwam.deploy.model_loader import delta_action_dims_from_cfg
from openwam.deploy.server import _load_deploy_yaml, build_server_from_config

ARM_DIMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
DELTA_SPEC = ["0-5", "7-12"]


class _Scheduler:
    num_train_timesteps = 1000

    def set_timesteps(self, n, shift=5.0):
        self.timesteps = self.sigmas = torch.linspace(1.0, 0.0, n)


class _ZeroOffsetArchitecture:
    """Stub model whose every 14-D action is zero: a zero offset for a delta checkpoint."""

    def __init__(self):
        self.video_scheduler = self.action_scheduler = _Scheduler()
        self.normalizer = None

    def load_checkpoint(self, path):
        del path

    def set_dtype_device(self, dtype, device):
        del dtype, device

    def eval(self):
        return self

    def attach_normalizer(self, normalizer):
        self.normalizer = normalizer

    def apply_compile_optimizations(self, compile_cfg):
        del compile_cfg

    def normalize_deploy_proprio(self, proprio):
        return torch.as_tensor(np.asarray(proprio, dtype=np.float32))

    def generate(self, **kwargs):
        del kwargs
        return {"video": None, "actions": np.zeros((32, 14), dtype=np.float32)}


@pytest.fixture()
def stub_model(monkeypatch):
    resolved = SimpleNamespace(
        params={}, registry_name="stub", canonical=SimpleNamespace(framework="dual_system", variant="stub")
    )
    monkeypatch.setattr(openwam.model, "resolve_architecture_config", lambda model_cfg: resolved)
    monkeypatch.setattr(openwam.model, "build_architecture", lambda name, params: _ZeroOffsetArchitecture())


def _ckpt_dir(tmp_path, delta_action_dims):
    dataloader = {"num_frames": 33, "normalize_mode": None, "action_mode": "joint"}
    if delta_action_dims is not None:
        dataloader["delta_action_dims"] = delta_action_dims
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    OmegaConf.save(
        OmegaConf.create({"model": {}, "dataloader": dataloader, "training": {"mixed_precision": "no"}}),
        ckpt / "config.yaml",
    )
    (ckpt / "checkpoint_step_1.safetensors").write_bytes(b"")
    return ckpt


def _serve(ckpt, *overrides):
    deploy_cfg = OmegaConf.merge(_load_deploy_yaml(), OmegaConf.from_dotlist(list(overrides)))
    return build_server_from_config(cfg=deploy_cfg, ckpt_dir=str(ckpt), device="cpu")


def test_delta_action_dims_from_cfg():
    assert delta_action_dims_from_cfg(OmegaConf.create({})) == ()
    assert delta_action_dims_from_cfg(OmegaConf.create({"dataloader": {"delta_action_dims": None}})) == ()
    assert delta_action_dims_from_cfg(OmegaConf.create({"dataloader": {"delta_action_dims": DELTA_SPEC}})) == ARM_DIMS


def test_delta_checkpoint_serves_absolute_targets(tmp_path, stub_model):
    server = _serve(_ckpt_dir(tmp_path, DELTA_SPEC))
    assert server.engine.architecture.delta_action_dims == ARM_DIMS
    state = np.linspace(-1.5, 1.5, 14).astype(np.float32)
    actions = server.engine.generate({"observation": {"state": state}})["actions"]
    # A zero offset holds the measured joints; the grippers stay the model's absolute output.
    np.testing.assert_allclose(actions[:, list(ARM_DIMS)], np.broadcast_to(state[list(ARM_DIMS)], (32, 12)))
    np.testing.assert_allclose(actions[:, [6, 13]], 0.0)


def test_absolute_checkpoint_is_untouched(tmp_path, stub_model):
    """A checkpoint from before delta actions (no key) serves the model output as is."""
    server = _serve(_ckpt_dir(tmp_path, None))
    assert server.engine.architecture.delta_action_dims == ()
    actions = server.engine.generate({"observation": {"state": np.ones(14, dtype=np.float32)}})["actions"]
    np.testing.assert_allclose(actions, 0.0)


def test_restating_the_checkpoint_dims_is_allowed(tmp_path, stub_model):
    server = _serve(_ckpt_dir(tmp_path, DELTA_SPEC), "dataloader.delta_action_dims=[0,1,2,3,4,5,7,8,9,10,11,12]")
    assert server.engine._delta_action_dims == ARM_DIMS


@pytest.mark.parametrize(
    "ckpt_dims, override",
    [
        (DELTA_SPEC, "dataloader.delta_action_dims=null"),  # disable: offsets would be absolute targets
        (None, 'dataloader.delta_action_dims=["0-5","7-12"]'),  # enable: absolute output + state
        (DELTA_SPEC, 'dataloader.delta_action_dims=["7-12"]'),  # change: left arm left relative
    ],
    ids=["disable", "enable", "change"],
)
def test_deploy_override_cannot_change_delta_dims(tmp_path, stub_model, ckpt_dims, override):
    with pytest.raises(ValueError, match="fixed by the checkpoint"):
        _serve(_ckpt_dir(tmp_path, ckpt_dims), override)
