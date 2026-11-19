"""Relational Geometry Forcing baseline (openwam/model/rgf_align.py, training.rgf)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from openwam.model.rgf_align import (
    RGF_PROJECTOR_FILE_PREFIX,
    RGFCacheReader,
    exclude_rgf_from_state_dict,
    latent_frame_positions,
    rgf_loss,
    rgf_projector_path,
    save_rgf_projector,
    view_token_indices,
)


def test_view_slots_partition_the_canvas_grid():
    idx = view_token_indices()
    assert [len(i) for i in idx] == [80, 20, 20]
    allidx = torch.cat(idx)
    assert sorted(allidx.tolist()) == list(range(120))
    # left wrist = rows 8-11, cols 0-4 of the 12x10 grid
    assert idx[1].tolist()[:5] == [80, 81, 82, 83, 84] and idx[1].tolist()[5] == 90


def test_latent_frame_positions():
    assert latent_frame_positions(9).tolist() == [0, 4, 8]
    assert latent_frame_positions(1).tolist() == [0]
    with pytest.raises(ValueError):
        latent_frame_positions(8)


def _rand_inputs(b=2, f=3, t=6, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.randn(b, f, 120, d, generator=g)
    scene = torch.randn(b, t, 3, d, generator=g)
    return grid, scene


def test_loss_is_zero_when_relations_match_and_scale_invariant():
    grid, scene = _rand_inputs()
    vv = torch.ones(2, 3, dtype=torch.bool)
    av = torch.ones(2, 6, dtype=torch.bool)
    out = rgf_loss(grid * 3.0, grid, vv, scene.reshape(2, 6, -1) * 0.5, scene, av, view_token_indices())
    for k, v in out.items():
        assert v.abs().item() < 1e-6, k
    other, oscene = _rand_inputs(seed=1)
    out2 = rgf_loss(other, grid, vv, oscene.reshape(2, 6, -1), scene, av, view_token_indices())
    assert all(v.item() > 1e-3 for v in out2.values())


def test_invalid_frames_and_padded_steps_do_not_count():
    grid, scene = _rand_inputs()
    feat, sfeat = _rand_inputs(seed=3)
    vv = torch.tensor([[True, True, False], [True, True, True]])
    av = torch.ones(2, 6, dtype=torch.bool)
    av[0, 4:] = False
    za = sfeat.reshape(2, 6, -1)
    base = rgf_loss(feat, grid, vv, za, scene, av, view_token_indices())
    feat2, za2 = feat.clone(), za.clone()
    feat2[0, 2] = torch.randn_like(feat2[0, 2]) * 10  # invalid latent frame
    za2[0, 4:] = torch.randn_like(za2[0, 4:]) * 10  # padded action steps
    moved = rgf_loss(feat2, grid, vv, za2, scene, av, view_token_indices())
    for k in base:
        assert torch.allclose(base[k], moved[k]), k


def test_values_match_a_hand_computation():
    # one view of 2 tokens, 1 latent frame, 2 action steps: every term is the Huber error of one cosine
    s = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])  # student tokens, cos = 1/sqrt2
    g = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]])  # teacher tokens, cos = -1
    e = 1 / 2**0.5 + 1  # |difference| > beta=1 -> |e| - 0.5
    want = e - 0.5
    out = rgf_loss(s, g, None, s[:, 0], g[:, 0].unsqueeze(2), None, [torch.tensor([0, 1])])
    assert out["intra"].item() == pytest.approx(want, rel=1e-5)
    assert out["temporal"].item() == pytest.approx(want, rel=1e-5)
    assert out["cross"].item() == 0.0  # a single view has no view pairs
    # small residual -> quadratic branch: 0.5 e^2
    g2 = torch.tensor([[[[1.0, 0.0], [1.0, 0.8]]]])
    e2 = 1 / 2**0.5 - float(F.cosine_similarity(g2[0, 0, 0], g2[0, 0, 1], dim=0))
    out2 = rgf_loss(s, g2, None, s[:, 0], g2[:, 0].unsqueeze(2), None, [torch.tensor([0, 1])])
    assert out2["intra"].item() == pytest.approx(0.5 * e2**2, rel=1e-4)


def test_cache_reader_indexing(tmp_path):
    t, d = 7, 4
    grid = np.arange(t, dtype=np.float16)[:, None, None] * np.ones((t, 120, d), np.float16)
    pool = np.arange(t, dtype=np.float16)[:, None, None] * np.ones((t, 3, d), np.float16)
    np.save(tmp_path / "episode_000003_grid.npy", grid)
    np.save(tmp_path / "episode_000003_pool.npy", pool)
    (tmp_path / "meta.json").write_text(json.dumps({"grid": [12, 10], "dim": d}))
    r = RGFCacheReader(str(tmp_path))
    assert r.grid(3, [0, 4, 9])[:, 0, 0].tolist() == [0, 4, 6]  # past the end repeats the last frame
    assert r.pool(3, [2, 3]).shape == (2, 3, d)


def test_heads_are_kept_out_of_checkpoints_and_saved_separately(tmp_path):
    from safetensors import safe_open

    from openwam.model.rgf_align import RGFConfig, RGFProjector

    heads = RGFProjector(video_dim=8, action_dim=4, hidden_dim=6, proj_dim=5)
    sd = {"video_backbone.dit.x": torch.zeros(1), **{f"rgf_projector.{k}": v for k, v in heads.state_dict().items()}}
    assert list(exclude_rgf_from_state_dict(sd)) == ["video_backbone.dit.x"]

    class _M:
        _rgf_cfg = RGFConfig()

    p = rgf_projector_path(str(tmp_path), 40)
    assert p.endswith(f"{RGF_PROJECTOR_FILE_PREFIX}40.safetensors") and "checkpoint_step_" not in p
    assert save_rgf_projector(p, sd, _M(), 40) == 8
    with safe_open(p, "pt") as f:
        assert set(f.keys()) == set(heads.state_dict()) and f.metadata()["global_step"] == "40"


def _tiny_driver(num_layers=4, dim=32, heads=4):
    from openwam.model.action_backbone.separate_action_dit import ActionDiT
    from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
    from tests.test_openwam_trainer import _MockVideoBackbone

    vb = _MockVideoBackbone(dim=dim, num_layers=num_layers, num_heads=heads)
    ab = ActionDiT(
        action_dim=7,
        dim=dim,
        ffn_dim=2 * dim,
        num_heads=heads,
        num_layers=num_layers,
        video_dim=dim,
        bridge_layers=tuple(range(num_layers)),
        variant="joint_self_attn",
        text_dim=dim,
    )
    return vb, ab, DualSystemMoTDriver(vb, ab, mot_checkpoint_mixed_attn=False)


@torch.no_grad()
def test_driver_captures_action_states_after_the_same_block():
    torch.manual_seed(0)
    vb, ab, drv = _tiny_driver()
    vstate = vb.prepare(latents=torch.zeros(2, 4, 3, 2, 2))
    astate = ab.prepare_state(
        torch.randn(2, 5, 7),
        torch.rand(2),
        context=torch.randn(2, 3, 32),
        context_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    import copy

    v0, a0 = copy.deepcopy(vstate), copy.deepcopy(astate)
    drv.capture_layers = (2,)
    drv.run_joint_loop(vstate, astate)
    # manual reference: run the first two steps with the same pre-built mask
    mask = drv._build_attention_mask(s_video=12, s_action=5, video_tokens_per_frame=4, device=torch.device("cpu"))
    for layer in range(2):
        v0, a0 = drv.step(layer, v0, a0, attn_mask=mask)
    assert torch.allclose(drv.captured[2], v0.hidden_states)
    assert torch.allclose(drv.captured_action[2], a0.payload.x_action)


def test_compute_loss_end_to_end_with_rgf():
    """Tiny dual-system architecture on CPU: the RGF terms are added with their weight and train the heads."""
    from openwam.model.rgf_align import RGFConfig
    from tests.test_architecture_variants import _build_arch, _run_compute_loss  # noqa: F401
    from tests.test_openwam_trainer import _make_fake_loss_inputs

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "bridge_layers": None,
        "bridge_interval": 1,
        "attention_mask_mode": "mutual",
    }
    arch = _build_arch("dual_system_self_attn", cfg, num_layers=4)
    # the mock's prediction is a fixed 8x8 grid; match the 12x10 canvas latent grid used here
    arch.video_backbone.finalize = lambda st: torch.randn(
        st.hidden_states.shape[0], 16, st.grid_frames, st.grid_height, st.grid_width
    )
    arch.enable_rgf(RGFConfig(layer=2, proj_dim=16, hidden_dim=16))
    arch.init_training_schedulers(1000)
    torch.manual_seed(0)
    t_action = 32
    inputs = _make_fake_loss_inputs(B=2, action_dim=7, T_action=t_action, video_dim=64)
    inputs["input_latents"] = torch.randn(2, 16, 3, 12, 10)  # 3 latent frames on the 12x10 canvas grid
    inputs["context"] = torch.randn(2, 4, arch.action_backbone.text_dim)
    inputs["context_mask"] = torch.ones(2, 4, dtype=torch.bool)
    inputs["seq_lens"] = torch.tensor([4, 4])
    inputs["rgf_grid"] = torch.randn(2, 3, 120, 24).half()
    inputs["rgf_grid_valid"] = torch.tensor([[True, True, True], [True, True, False]])
    inputs["rgf_pool"] = torch.randn(2, t_action, 3, 24).half()
    inputs["rgf_pool_valid"] = torch.ones(2, t_action, dtype=torch.bool)
    if arch.uses_proprioception:
        inputs["proprio"] = torch.randn(2, int(arch.action_backbone.state_dim))
    out = arch.compute_loss(**inputs, actions=torch.randn(2, t_action, 7))
    for k in ("loss_rgf", "loss_rgf_intra", "loss_rgf_cross", "loss_rgf_temporal"):
        assert torch.isfinite(out[k]) and out[k].item() > 0, k
    assert out["loss"].item() == pytest.approx(
        (out["loss_video"] + out["loss_action"] + 0.5 * out["loss_rgf"]).item(), rel=1e-5
    )
    out["loss"].backward()
    for name in ("video", "action"):
        g = getattr(arch.rgf_projector, name)[0].weight.grad
        assert g is not None and g.abs().sum() > 0, name
    # the deploy checkpoint keeps none of it
    assert not [k for k in exclude_rgf_from_state_dict(arch.state_dict()) if k.startswith("rgf_projector.")]
