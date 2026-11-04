"""Open-loop evaluator tests on a synthetic YAM LeRobot-v2.1 bucket with oracle policies.

Video decoding is mocked: every frame is a solid colour encoding (camera, step),
so an oracle that knows the recording can reproduce canvases exactly. An oracle
must score zero action error and maximal PSNR; a policy one step out of phase
must not, which pins the alignment the evaluator scores at.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.eval import open_loop
from openwam.eval.open_loop import (
    MAX_PSNR_DB,
    chunk_starts,
    dimension_groups,
    evaluate,
    frame_steps,
    psnr,
    read_episode,
    rollout_episode,
)
from tests.dataloader.test_yam import CAMERA_COLORS, LENGTHS, _episode_action, _episode_state, make_bucket

CAMERAS = tuple(CAMERA_COLORS)


def _solid(camera: str, step: int, size=(640, 480)) -> Image.Image:
    return Image.new("RGB", size, (step % 256, 60 * CAMERAS.index(camera), 128))


def _canvas_at(step: int) -> Image.Image:
    return assemble_multiview_layout({c: _solid(c, step) for c in CAMERAS}, list(CAMERAS), 384, 320)


@pytest.fixture(autouse=True)
def _mock_decode(monkeypatch):
    def fake_decode(path, frame_indices, height, width):
        camera = next(c for c in CAMERAS if f"/{c}/" in str(path))
        return [_solid(camera, i, (width, height)) for i in frame_indices]

    monkeypatch.setattr(open_loop, "decode_video_frames", fake_decode)


class OraclePolicy:
    """Knows the recording: finds the step from the state and returns what was recorded next."""

    cameras = CAMERAS
    chunk_len = 32
    video_stride = 4

    def __init__(self, shift: int = 0):
        self.shift = shift
        self.where = {}
        for ep, length in enumerate(LENGTHS):
            for t, row in enumerate(_episode_state(ep, length).astype(np.float64)):
                self.where[row.tobytes()] = (ep, t)

    def predict(self, images, state, prompt):
        ep, t = self.where[np.asarray(state, dtype=np.float64).tobytes()]
        actions = _episode_action(ep, LENGTHS[ep]).astype(np.float64)
        idx = np.clip(np.arange(t, t + self.chunk_len) + self.shift, 0, len(actions) - 1)
        video = [_canvas_at(t + j * self.video_stride) for j in range(self.chunk_len // self.video_stride + 1)]
        return actions[idx], video

    def canvas(self, images, state):
        return assemble_multiview_layout(dict(images), list(self.cameras), 384, 320)


def test_chunk_and_frame_steps():
    assert chunk_starts(40, 32) == [0, 32]
    assert frame_steps(40, 32, 32, 4) == [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
    assert frame_steps(40, 32, 32, None) == [0, 32]


def test_dimension_groups_split_arms_and_grippers():
    from openwam.dataloader.yam import JOINT_NAMES

    assert dimension_groups(JOINT_NAMES) == {
        "left joints": [0, 1, 2, 3, 4, 5],
        "grippers": [6, 13],
        "right joints": [7, 8, 9, 10, 11, 12],
    }


def test_psnr_caps_identical_images():
    image = _canvas_at(3)
    assert psnr(image, image) == MAX_PSNR_DB
    assert psnr(image, _canvas_at(4)) < MAX_PSNR_DB


def test_rollout_stitches_executed_actions_and_hold_baseline(tmp_path):
    bucket = make_bucket(tmp_path)
    episode = read_episode(bucket, 0, CAMERAS)
    episode.load_frames(chunk_starts(len(episode), 8))
    rollout = rollout_episode(OraclePolicy(), episode, horizon=8)
    assert rollout.starts == [0, 8, 16, 24, 32]
    np.testing.assert_allclose(rollout.executed(), episode.actions)
    hold = rollout.hold()
    for start in rollout.starts:
        np.testing.assert_allclose(
            hold[start : start + 8], np.broadcast_to(episode.states[start], (8, 14))[: 40 - start]
        )


def test_wrong_chunk_shape_is_rejected(tmp_path):
    class Short(OraclePolicy):
        def predict(self, images, state, prompt):
            actions, video = super().predict(images, state, prompt)
            return actions[:-1], video

    bucket = make_bucket(tmp_path)
    episode = read_episode(bucket, 0, CAMERAS)
    episode.load_frames([0, 32])
    with pytest.raises(ValueError, match="expected"):
        rollout_episode(Short(), episode, horizon=32)


def test_oracle_scores_perfectly_and_writes_outputs(tmp_path):
    pytest.importorskip("matplotlib")
    import imageio.v2 as imageio

    bucket = make_bucket(tmp_path)
    out = tmp_path / "eval"
    summary = evaluate(OraclePolicy(), bucket, [0, 2], out, horizon=32)

    assert all(v == 0 for v in summary["overall"]["mae"].values())
    assert summary["overall"]["hold_mae"]["right joints"] > 0
    first = summary["per_episode"][0]
    assert (first["steps"], first["chunks"]) == (LENGTHS[0], 2)
    assert first["video_psnr"] == first["recon_psnr"] == MAX_PSNR_DB
    assert set(summary["overall"]["video_psnr_by_offset"]) == set(range(0, 33, 4))

    for stem in ("episode_000000", "episode_000002"):
        assert (out / f"{stem}_actions.png").stat().st_size > 0
        arrays = np.load(out / f"{stem}.npz")
        np.testing.assert_allclose(arrays["executed"], arrays["recorded"])
    # Every generated frame is in the video, including those past the episode end.
    assert imageio.get_reader(out / "episode_000000_video.mp4").count_frames() == 2 * 9
    assert (out / "summary.json").is_file() and (out / "summary.png").is_file()


def test_one_step_misalignment_is_not_forgiven(tmp_path):
    pytest.importorskip("matplotlib")
    bucket = make_bucket(tmp_path)
    summary = evaluate(OraclePolicy(shift=1), bucket, [1], tmp_path / "eval", horizon=32, video=False)
    assert summary["overall"]["mae"]["right joints"] > 0
    assert summary["overall"]["video_psnr_by_offset"] == {}


def test_eval_dir_mirrors_the_checkpoint_dir_under_eval_out():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "open_loop_eval_script", Path(__file__).resolve().parents[1] / "scripts" / "open_loop_eval.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    outputs = script.PROJECT_ROOT / "outputs"
    run = Path("place_cube_right_n56/2026-10-23_22-13-11")
    assert script.eval_dir(outputs / "openwam_checkpoints" / run, "checkpoint_step_4080.safetensors") == (
        outputs / "eval_out" / run / "checkpoint_step_4080" / "open_loop"
    )
    assert script.eval_dir(Path("/elsewhere/my_run"), "step_9.safetensors") == (
        outputs / "eval_out" / "my_run" / "step_9" / "open_loop"
    )
