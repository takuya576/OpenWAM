"""GAE alignment loss (training.gae_align): projector layout, loss, checkpoint exclusion."""

import os
import tempfile
from types import SimpleNamespace

import torch

from openwam.model.gae_align import (
    GAEAlignConfig,
    GAEProjector,
    exclude_gae_from_state_dict,
    gae_align_loss,
    load_projector,
    projector_path,
    save_projector,
)


def _tokens(dim=96):
    torch.manual_seed(0)
    return torch.randn(2, 2, 12, 10, dim, requires_grad=True)


def test_projector_shape_and_layout():
    proj = GAEProjector(in_dim=96, hidden_dim=64)
    tok = _tokens()
    out = proj(tok)
    assert out.shape == (2, 8, 3, 64, 32, 40)
    # Each output location depends on exactly one token: (sample, latent frame, token row, token col).
    # frame 5 -> latent frame 1; right wrist (slot 2), output row 9 -> token row 8 + 9 // 8, col 30 -> 5 + 30 // 8
    g = torch.autograd.grad(out[0, 5, 2, :, 9, 30].sum(), tok, retain_graph=True)[0]
    assert g.abs().sum(-1).nonzero().tolist() == [[0, 1, 9, 8]]
    # frame 2 -> latent frame 0; head (slot 0), output row 13 -> token row 13 // 4, col 22 -> 22 // 4
    g = torch.autograd.grad(out[1, 2, 0, :, 13, 22].sum(), tok)[0]
    assert g.abs().sum(-1).nonzero().tolist() == [[1, 0, 3, 5]]


def test_loss_zero_for_perfect_prediction_and_masks_invalid_frames():
    torch.manual_seed(0)
    target = torch.randn(2, 8, 3, 64, 27, 36).half()
    perfect = torch.nn.functional.interpolate(
        target.reshape(-1, 64, 27, 36).float(), size=(32, 40), mode="bilinear", align_corners=False
    ).reshape(2, 8, 3, 64, 32, 40)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[1, 5:] = False
    assert float(gae_align_loss(perfect, target, valid, "cosine")) < 1e-5
    assert float(gae_align_loss(perfect, target, None, "mse")) < 1e-8
    wrong_where_invalid = perfect.clone()
    wrong_where_invalid[1, 5:] += 5.0
    assert float(gae_align_loss(wrong_where_invalid, target, valid, "cosine_mse")) < 1e-5
    assert float(gae_align_loss(wrong_where_invalid, target, None, "cosine_mse")) > 0.1


def test_projector_is_not_saved():
    sd = {"gae_projector.inp.weight": torch.zeros(1), "video_backbone.x": torch.zeros(1)}
    assert list(exclude_gae_from_state_dict(sd)) == ["video_backbone.x"]


def test_projector_file_round_trip_and_invisible_to_deploy_lookup():
    from openwam.deploy.model_loader import _find_latest_checkpoint as find_latest_checkpoint
    from openwam.train.utils.checkpointing import find_latest_weights

    torch.manual_seed(0)
    proj = GAEProjector(in_dim=96, hidden_dim=64).to(torch.bfloat16)
    model = SimpleNamespace(gae_projector=proj, _gae_align_cfg=GAEAlignConfig(layer=15, weight=0.5, loss="cosine"))
    state_dict = {"video_backbone.x": torch.zeros(2)}
    state_dict.update({f"gae_projector.{k}": v for k, v in proj.state_dict().items()})
    with tempfile.TemporaryDirectory() as d:
        torch.save({}, os.path.join(d, "checkpoint_step_20.safetensors"))  # stand-in deploy file
        path = projector_path(d, 20)
        assert save_projector(path, state_dict, model, 20) == len(proj.state_dict())
        loaded, info = load_projector(path)
        assert info["global_step"] == 20 and info["align"] == {"layer": 15, "weight": 0.5, "loss": "cosine"}
        for k, v in proj.state_dict().items():
            assert torch.equal(loaded.state_dict()[k].to(v.dtype), v), k
        tok = torch.randn(1, 2, 12, 10, 96)
        assert torch.allclose(loaded.to(torch.float32)(tok), proj.float()(tok))
        # Deploy and finetune lookups still pick the deploy checkpoint, never the projector.
        assert os.path.basename(find_latest_weights(d)) == "checkpoint_step_20.safetensors"
        assert os.path.basename(find_latest_checkpoint(d)) == "checkpoint_step_20.safetensors"


def test_projector_files_pruned_with_their_checkpoints():
    from openwam.train.utils.checkpointing import finalize_keep_weights_only, manage_checkpoints

    with tempfile.TemporaryDirectory() as d:
        for step in (10, 20, 30):
            for name in (f"checkpoint_step_{step}.safetensors", f"gae_projector_step_{step}.safetensors"):
                open(os.path.join(d, name), "w").close()
        manage_checkpoints(d, keep_last_k=2)
        assert sorted(os.listdir(d)) == [
            "checkpoint_step_20.safetensors",
            "checkpoint_step_30.safetensors",
            "gae_projector_step_20.safetensors",
            "gae_projector_step_30.safetensors",
        ]
        finalize_keep_weights_only(d, keep_last_k=1)
        assert sorted(os.listdir(d)) == ["checkpoint_step_30.safetensors", "gae_projector_step_30.safetensors"]


if __name__ == "__main__":
    test_projector_shape_and_layout()
    test_loss_zero_for_perfect_prediction_and_masks_invalid_frames()
    test_projector_is_not_saved()
    test_projector_file_round_trip_and_invisible_to_deploy_lookup()
    test_projector_files_pruned_with_their_checkpoints()
    print("ok")
