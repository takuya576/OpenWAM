"""Real-robot YAM client (scripts/run_yam_policy.py) against a live PolicyServer and a fake rd serve.

The policy server is the real one, serving a scripted engine over a real
WebSocket, so the ``chunk`` round trip, image composition and proprio
passthrough are exercised as deployed. ``chiral.PolicyClient`` is replaced by a
fake robot that tracks its commands perfectly (or not at all), recording every
command it is sent.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from openwam.deploy.server import PolicyServer

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_yam_policy", REPO / "scripts" / "run_yam_policy.py")
ryp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ryp)

CHUNK_LEN = 32
HZ = 600.0  # fast enough that a 32-step chunk dispatches in ~50 ms
HOME = np.array([0.0] * 6 + [1.0] + [0.0] * 6 + [1.0])
COLORS = {"scene_camera": (200, 0, 0), "left_wrist_camera": (0, 200, 0), "right_wrist_camera": (0, 0, 200)}


# --- scripted policy server ---------------------------------------------------------


class ScriptedEngine:
    """Returns ``chunk_fn(state)``; records each request's conditions.

    With ``decode_video`` it "imagines" the input canvas unchanged for all 9 frames,
    so against a robot whose cameras never change every frame should match the live
    canvas exactly.
    """

    architecture = None

    def __init__(self):
        self.calls = []
        self.chunk_fn = lambda state: np.tile(state, (CHUNK_LEN, 1))

    def generate(self, conditions):
        self.calls.append(conditions)
        video = [conditions["first_frame_image"][0]] * 9 if conditions.get("decode_video") else None
        return {"actions": self.chunk_fn(np.asarray(conditions["proprio"], dtype=np.float64)), "video": video}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def served():
    engine = ScriptedEngine()
    cfg = OmegaConf.create(
        {
            "dataloader": {
                "multiview": True,
                "camera_layout": [
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                ],
                "height": 384,
                "width": 320,
            },
            "inference": {"inference_mode": "sync"},
            "model": {"architecture": {"use_proprioception": True}},
        }
    )
    server = PolicyServer(engine=engine, cfg=cfg)
    server._ckpt_contract = lambda: {"representation": "joint"}
    port = _free_port()
    threading.Thread(target=server.run, kwargs={"host": "127.0.0.1", "port": port}, daemon=True).start()
    deadline = time.monotonic() + 10
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    return SimpleNamespace(engine=engine, uri=f"ws://127.0.0.1:{port}")


@pytest.fixture
def engine(served):
    served.engine.calls.clear()
    served.engine.chunk_fn = lambda state: np.tile(state, (CHUNK_LEN, 1))
    return served.engine


# --- fake rd serve --------------------------------------------------------------------


def _frame(rgb, size=(480, 640)):
    return np.full((*size, 3), rgb, dtype=np.uint8)


class FakeRobot:
    """The subset of chiral's PolicyClient the client uses. ``follows=False`` never moves."""

    def __init__(self, follows=True, **meta):
        self.q = HOME.copy()
        self.follows = follows
        self.sent = []
        self.closed = False
        self._action_queue = deque()
        self.meta = {"action_type": "joint", "action_shape": [14], "control_hz": HZ, "max_joint_delta": 0.35, **meta}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def _obs(self):
        cameras = [SimpleNamespace(name=n, image=_frame(c), depth=None) for n, c in COLORS.items()]
        proprios = {"follower_l_joint_pos": self.q[:7].copy(), "follower_r_joint_pos": self.q[7:].copy()}
        return SimpleNamespace(cameras=cameras, proprios=proprios)

    def get_metadata(self):
        return dict(self.meta, cameras=list(COLORS))

    def reset(self):
        self.q = HOME.copy()
        return self._obs(), {}

    def get_obs(self):
        return self._obs()

    @property
    def latest_obs(self):
        return self._obs()

    def put_action(self, action):
        self.sent.append(np.array(action, dtype=np.float64))
        if self.follows:
            self.q = np.array(action, dtype=np.float64)

    def start_obs_stream(self, hz):
        pass

    def stop_obs_stream(self):
        pass

    def start_action_dispatch(self, hz):
        pass

    def stop_action_dispatch(self):
        pass


class LaggingRobot(FakeRobot):
    """Arms that approach their latest command with time constant ``tau`` seconds, so they trail a moving command."""

    def __init__(self, tau=0.02, **meta):
        super().__init__(**meta)
        self.tau, self.target, self.t = tau, HOME.copy(), time.perf_counter()

    def _advance(self):
        now = time.perf_counter()
        self.q = self.target + (self.q - self.target) * np.exp(-(now - self.t) / self.tau)
        self.t = now

    def _obs(self):
        self._advance()
        return super()._obs()

    def reset(self):
        self.target = HOME.copy()
        return super().reset()

    def put_action(self, action):
        self._advance()
        self.sent.append(np.array(action, dtype=np.float64))
        self.target = np.array(action, dtype=np.float64)


@pytest.fixture
def run(served, monkeypatch, tmp_path):
    """``run(robot, *flags) -> (exit code, ep<N>_summary.json, ep<N>_rollout.npz)`` of the rollout it ran."""
    previous = signal.getsignal(signal.SIGINT)

    def _run(robot, *flags):
        monkeypatch.setitem(sys.modules, "chiral", SimpleNamespace(PolicyClient=lambda uri: robot))
        out = tmp_path / "rollout"
        episode = ryp.next_episode(out)
        argv = ["--prompt", "place the cube", "--robot-uri", "ws://fake", "--policy-uri", served.uri]
        argv += ["--control-hz", str(HZ), "--out", str(out), *flags]
        code = ryp.main(argv)
        summary = json.loads((out / f"ep{episode}_summary.json").read_text())
        return code, summary, np.load(out / f"ep{episode}_rollout.npz")

    yield _run
    signal.signal(signal.SIGINT, previous)


def _ramp(rate, dims=(7,)):
    """Chunk moving ``dims`` by ``rate`` rad per step from the measured state."""

    def fn(state):
        chunk = np.tile(state, (CHUNK_LEN, 1))
        chunk[:, list(dims)] += rate * np.arange(1, CHUNK_LEN + 1)[:, None]
        return chunk

    return fn


# --- end to end -------------------------------------------------------------------------


def test_dry_run_requests_chunks_and_sends_nothing(run, engine, tmp_path):
    robot = FakeRobot()
    code, summary, npz = run(robot, "--dry-run", "--max-chunks", "2")
    assert code == 0 and robot.sent == [] and robot.closed
    assert summary["halt_reason"] is None and (summary["chunks"], summary["steps"]) == (2, 2 * CHUNK_LEN)
    assert len(engine.calls) == 3  # warmup + 2 chunks
    call = engine.calls[-1]
    assert call["prompt"] == "place the cube"
    np.testing.assert_allclose(call["proprio"], HOME)
    canvas = call["first_frame_image"][0]
    assert canvas.size == (320, 384)
    # head on top, left wrist bottom-left, right wrist bottom-right
    assert canvas.getpixel((160, 128)) == COLORS["scene_camera"]
    assert canvas.getpixel((80, 320)) == COLORS["left_wrist_camera"]
    assert canvas.getpixel((240, 320)) == COLORS["right_wrist_camera"]
    assert npz["predicted"].shape == (2, CHUNK_LEN, 14) and npz["commanded"].shape == (2 * CHUNK_LEN, 14)
    assert "decode_video" not in call and summary["video_psnr_db"] is None
    assert summary["episode"] == 0 and not (tmp_path / "rollout" / "ep0_imagined_vs_live.mp4").exists()


def test_rollout_is_filed_under_the_served_checkpoint(served, engine, monkeypatch, tmp_path):
    run = Path("place_cube_right_n56/2026-10-23_22-13-11")
    weights = tmp_path / "outputs/openwam_checkpoints" / run / "checkpoint_step_4080.safetensors"
    monkeypatch.setattr(engine, "architecture", SimpleNamespace(ckpt_path=str(weights)), raising=False)
    monkeypatch.setitem(sys.modules, "chiral", SimpleNamespace(PolicyClient=lambda uri: FakeRobot()))
    monkeypatch.chdir(tmp_path)
    previous = signal.getsignal(signal.SIGINT)
    try:
        argv = ["--prompt", "p", "--robot-uri", "ws://fake", "--policy-uri", served.uri, "--control-hz", str(HZ)]
        for _ in range(2):  # a second session numbers on instead of overwriting the first
            assert ryp.main([*argv, "--dry-run", "--max-chunks", "1"]) == 0
    finally:
        signal.signal(signal.SIGINT, previous)
    out = tmp_path / "outputs/eval_out" / run / "checkpoint_step_4080" / "closed_loop"
    assert sorted(p.name for p in out.iterdir() if p.suffix != ".png") == [
        f"ep{n}_{name}" for n in (0, 1) for name in ("rollout.npz", "summary.json")
    ]
    summary = json.loads((out / "ep1_summary.json").read_text())
    assert summary["episode"] == 1 and summary["policy"]["checkpoint"] == str(weights)


def test_closed_loop_dir_outside_the_checkpoint_root_or_unreported():
    assert ryp.closed_loop_dir("/elsewhere/my_run/step_9.safetensors") == Path(
        "outputs/eval_out/my_run/step_9/closed_loop"
    )
    assert ryp.closed_loop_dir(None) == Path("outputs/eval_out/unlinked/closed_loop")


def test_next_episode_and_success_tally_read_the_ep_files(tmp_path):
    assert ryp.next_episode(tmp_path / "missing") == 0
    for name, success in (("ep0", True), ("ep2", False), ("ep3", None), ("ep10", True)):
        (tmp_path / f"{name}_summary.json").write_text(json.dumps({"success": success}))
    (tmp_path / "ep11_trajectory.png").write_bytes(b"")  # a rollout killed before its summary
    (tmp_path / "notes_ep99.txt").write_text("")
    assert ryp.next_episode(tmp_path) == 12
    assert ryp.success_tally(tmp_path) == (2, 3)


def test_video_pairs_every_imagined_frame_with_the_live_canvas(run, engine, tmp_path):
    pytest.importorskip("matplotlib")
    engine.chunk_fn = _ramp(0.01)
    code, summary, _ = run(FakeRobot(), "--video", "--max-chunks", "2")
    assert code == 0 and summary["halt_reason"] is None
    assert all(call.get("decode_video") for call in engine.calls)  # warmup included
    # 9 imagined frames per chunk: input + one per 4 executed steps, all matching the unchanging cameras
    assert summary["video_psnr_db"] == {str(j): 100.0 for j in range(9)}
    out = tmp_path / "rollout"
    assert (out / "ep0_imagined_vs_live.mp4").stat().st_size > 0 and (out / "ep0_trajectory.png").stat().st_size > 0


def test_video_marks_imagined_frames_past_a_halt_as_not_executed(run, engine):
    engine.chunk_fn = _ramp(0.04)
    _, summary, _ = run(FakeRobot(follows=False), "--video", "--max-chunks", "2")
    assert "--max-tracking-error" in summary["halt_reason"]
    # halted at step 32 of the first chunk, before frame 8's live canvas was due 4 steps later
    assert summary["video_psnr_db"] == {str(j): 100.0 for j in range(8)}

    _, summary, _ = run(FakeRobot(follows=False), "--video", "--max-chunks", "2", "--live-video-delay", "0")
    assert summary["video_psnr_db"] == {str(j): 100.0 for j in range(9)}


def test_video_with_a_short_horizon_ends_each_chunk_on_its_last_live_canvas(run, engine, monkeypatch):
    clips, video_frames = [], ryp.video_frames

    def spy(record):
        clips.extend(record["video"])
        return video_frames(record)

    monkeypatch.setattr(ryp, "video_frames", spy)
    _, summary, _ = run(FakeRobot(), "--video", "--horizon", "10", "--max-chunks", "2")
    assert summary["halt_reason"] is None and summary["steps"] == 20
    # live canvases at steps 0, 4+4 and 8+4 of each chunk, then the one 4 steps after step 10, where it ended
    assert [(len(clip["live"]), clip["last"][0]) for clip in clips] == [(3, 14), (3, 14)]
    assert summary["video_psnr_db"] == {str(j): 100.0 for j in range(3)}


class CountingRobot(FakeRobot):
    """Its head camera shows, in the red channel, how many commands it has been sent."""

    def _obs(self):
        obs = super()._obs()
        obs.cameras[0].image = _frame((len(self.sent), 0, 0))
        return obs


def test_live_video_delay_captures_each_canvas_that_many_steps_late(run, engine, monkeypatch):
    clips, video_frames = [], ryp.video_frames

    def spy(record):
        clips.extend(record["video"])
        return video_frames(record)

    monkeypatch.setattr(ryp, "video_frames", spy)
    sent_at_capture = lambda clip: [int(canvas[0, 0, 0]) for canvas in clip["live"]]  # noqa: E731

    run(CountingRobot(), "--video", "--max-chunks", "1", "--live-video-delay", "0")
    assert sent_at_capture(clips.pop()) == list(range(0, 33, 4))
    # frame 8's canvas comes after the chunk's 32 commands, while the arms finish the last one
    run(CountingRobot(), "--video", "--max-chunks", "1")
    assert sent_at_capture(clips.pop()) == [0, *range(8, 33, 4), 32]


def test_video_frames_hold_the_last_live_canvas_past_the_executed_steps():
    size = (96, 128)
    gray, blue = _frame((10, 10, 10), size), _frame((0, 0, 250), size)
    clip = {"start": 32, "stride": 4, "size": size, "imagined": [ryp.encode_png(gray)] * 9, "live": [gray] * 3}
    width = size[1]
    live_px = lambda frame: tuple(frame[-2, 2 * width])  # noqa: E731 -- live panel's bottom-right, clear of labels
    diff_px = lambda frame: tuple(frame[-2, 3 * width + 2])  # noqa: E731

    frames, psnr = ryp.video_frames({"video": [{**clip, "last": (10, blue)}]})
    assert len(frames) == 9 and psnr == {0: 100.0, 1: 100.0, 2: 100.0}
    assert all(live_px(frame) == (10, 10, 10) for frame in frames[:3])
    assert all(live_px(frame) == (0, 0, 250) and diff_px(frame) == (0, 0, 0) for frame in frames[3:])

    # a horizon on the frame grid ends on a captured frame, which is the one held
    frames, _ = ryp.video_frames({"video": [{**clip, "live": [gray, gray, blue]}]})
    assert all(live_px(frame) == (0, 0, 250) for frame in frames[2:])


def test_settle_observes_the_arms_where_the_next_chunk_starts(run, engine):
    # Arms ~12 steps behind a 0.01 rad/step ramp, so ~0.12 rad behind the last command when a chunk ends.
    engine.chunk_fn = _ramp(0.01)
    _, plain, _ = run(LaggingRobot(), "--max-chunks", "3")
    border = plain["chunk_border_rad"]
    assert border["state_behind_last_command"]["p50"] > 0.05 and border["border_step"]["p50"] > 0.05
    assert plain["timing_ms"]["settle"] is None

    _, settled, _ = run(LaggingRobot(), "--settle", "--max-chunks", "3")
    border = settled["chunk_border_rad"]
    assert settled["halt_reason"] is None and settled["settle_timeouts"] == 0
    assert border["state_behind_last_command"]["max"] <= ryp.SETTLE_TOL_RAD
    assert border["border_step"]["max"] < 0.035  # one ramp step plus at most the settle tolerance
    assert settled["timing_ms"]["settle"]["max"] < 1e3 * ryp.SETTLE_TIMEOUT_S


def test_compose_canvas_matches_the_server_layout():
    from PIL import Image

    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    rng = np.random.default_rng(0)
    frames = {key: rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for key in ryp.IMAGE_KEYS}
    expected = assemble_multiview_layout(
        {key: Image.fromarray(f) for key, f in frames.items()}, list(ryp.IMAGE_KEYS), out_h=384, out_w=320
    )
    np.testing.assert_array_equal(ryp.compose_canvas(frames, 384, 320), np.asarray(expected))


def test_dry_run_slew_limits_from_the_unmoved_arms_not_from_unsent_commands(run, engine):
    engine.chunk_fn = _ramp(0.03)  # each chunk reaches 0.96 rad from where the arm is measured
    code, summary, _ = run(FakeRobot(), "--dry-run", "--max-chunks", "3")
    assert summary["halt_reason"] is None and summary["slew_clipped_steps"] == 0 and summary["chunks"] == 3


def test_live_rollout_dispatches_clipped_commands_and_holds_an_arm(run, engine):
    def fn(state):
        chunk = _ramp(0.01)(state)
        chunk[:, 0] += 0.05  # left arm wanders; --hold-arm left must pin it
        chunk[:, 13] = 1.3  # right gripper beyond fully open
        return chunk

    engine.chunk_fn = fn
    robot = FakeRobot()
    code, summary, npz = run(robot, "--hold-arm", "left", "--max-chunks", "2")
    sent = np.stack(robot.sent)
    assert code == 0 and summary["halt_reason"] is None and len(sent) == 2 * CHUNK_LEN
    np.testing.assert_allclose(sent[:, :7], np.tile(HOME[:7], (len(sent), 1)))
    assert np.all(sent[:, 13] == 1.0) and summary["clip"]["gripper_clipped"] == 2 * CHUNK_LEN
    # each chunk continues from the joints the robot reached under the previous one
    np.testing.assert_allclose(sent[:, 7], 0.01 * np.arange(1, 2 * CHUNK_LEN + 1), atol=1e-6)
    np.testing.assert_allclose(npz["chunk_states"][1, 7], sent[CHUNK_LEN - 1, 7], atol=1e-6)
    assert summary["tracking_rad"]["max"] < 1e-6


def test_command_far_from_measured_joints_halts_before_moving(run, engine):
    engine.chunk_fn = lambda state: np.tile(state + np.r_[np.zeros(7), 0.4, np.zeros(6)], (CHUNK_LEN, 1))
    robot = FakeRobot()
    code, summary, npz = run(robot)
    assert code == 0 and robot.sent == []
    assert "right_joint_1" in summary["halt_reason"] and "--max-command-gap" in summary["halt_reason"]
    assert npz["predicted"].shape == (1, CHUNK_LEN, 14)  # the refused chunk is kept


def test_arm_that_does_not_follow_halts_on_tracking_error(run, engine):
    engine.chunk_fn = _ramp(0.04)
    robot = FakeRobot(follows=False)
    code, summary, _ = run(robot, "--max-chunks", "3")
    assert "--max-tracking-error" in summary["halt_reason"]
    # halted inside the first chunk, once the closest of the last 25 commands was 0.32 rad away
    assert len(robot.sent) == CHUNK_LEN and summary["steps"] == CHUNK_LEN


def test_out_of_range_gripper_state_halts_before_moving(run, engine):
    robot = FakeRobot()
    robot.reset = lambda: (robot._obs(), {})
    robot.q[6] = 45.0  # gripper reported in motor units, not i2rt's [0, 1]
    code, summary, _ = run(robot)
    assert robot.sent == [] and "outside [0, 1]" in summary["halt_reason"]


def test_robot_server_in_eef_mode_is_refused(run, engine):
    robot = FakeRobot(action_type="ee_pose", action_shape=[16])
    with pytest.raises(SystemExit, match="--action-type joint"):
        run(robot)
    assert robot.sent == [] and robot.closed


# --- pieces ------------------------------------------------------------------------------


def test_measured_joints_is_left_then_right():
    obs = SimpleNamespace(proprios={"follower_l_joint_pos": np.arange(7), "follower_r_joint_pos": np.arange(7, 14)})
    np.testing.assert_array_equal(ryp.measured_joints(obs), np.arange(14))
    with pytest.raises(RuntimeError, match="follower_r_joint_pos"):
        ryp.measured_joints(SimpleNamespace(proprios={"follower_l_joint_pos": np.arange(7)}))


def test_camera_frames_maps_names_and_passes_frames_through():
    top_bottom = np.zeros((4, 6, 3), np.uint8)
    top_bottom[:2] = 255
    obs = SimpleNamespace(
        cameras=[SimpleNamespace(name=n, image=top_bottom) for n in ("cam_a", "cam_b", "cam_c", "depth_only")]
    )
    cameras = dict(zip(("cam_a", "cam_b", "cam_c"), ryp.IMAGE_KEYS))
    frames = ryp.camera_frames(obs, cameras)
    assert set(frames) == set(ryp.IMAGE_KEYS)
    for frame in frames.values():
        np.testing.assert_array_equal(frame, top_bottom)
    with pytest.raises(RuntimeError, match="--cameras"):
        ryp.camera_frames(obs, {"cam_a": "head_camera", "missing": "left_wrist_camera"})


def test_clip_to_limits_counts_what_it_changed():
    cmds = np.tile(HOME, (3, 1))
    cmds[1, 1] = -0.5  # left joint_2 below its widened lower limit of -0.15
    cmds[2, 13] = -0.2
    out, stats = ryp.clip_to_limits(cmds)
    assert out[1, 1] == pytest.approx(-0.15) and out[2, 13] == 0.0
    assert stats == {"joint_clipped_steps": 1, "max_joint_overshoot": pytest.approx(0.35), "gripper_clipped": 1}


def test_slew_limit_bounds_arm_joints_but_not_grippers():
    cmds = np.tile(HOME, (3, 1))
    cmds[:, 7] = 1.0
    cmds[:, 13] = 0.0
    out, stats = ryp.slew_limit(cmds, HOME, max_step=0.35)
    np.testing.assert_allclose(out[:, 7], [0.35, 0.70, 1.0])
    np.testing.assert_allclose(out[:, 13], 0.0)
    assert stats["clipped_steps"] == 2 and stats["max_deviation"] == pytest.approx(0.65)


def test_tracking_error_finds_the_lag_and_reports_none_while_dwelling():
    history = np.tile(HOME, (10, 1))
    history[:, 7] = 0.1 * np.arange(10)
    measured = history[4].copy()
    error, lag = ryp.tracking_error(history, measured)
    assert error == pytest.approx(0.0) and lag == 5
    error, lag = ryp.tracking_error(np.tile(HOME, (10, 1)), HOME)
    assert error == 0.0 and lag is None


def test_settled_obs_arrival_rest_and_timeout(monkeypatch):
    monkeypatch.setattr(ryp, "SETTLE_STILL_S", 0.05)
    monkeypatch.setattr(ryp, "SETTLE_TIMEOUT_S", 0.15)
    running = SimpleNamespace(stop=False)

    # arrived: arm joints on the command; a gripper closed on an object never is, and does not count
    target = HOME.copy()
    target[13] = 0.0
    obs, waited, settled = ryp.settled_obs(FakeRobot(), target, running)
    assert settled and waited < 0.05 and obs is not None

    # at rest short of the command, as under load: settled once still for SETTLE_STILL_S
    target = HOME.copy()
    target[7] += 0.1
    _, waited, settled = ryp.settled_obs(FakeRobot(follows=False), target, running)
    assert settled and 0.05 <= waited < 0.15

    # still moving and never arriving: the latest observation after SETTLE_TIMEOUT_S
    class Shaking(FakeRobot):
        @property
        def latest_obs(self):
            self.q[7] = 0.1 - self.q[7]
            return self._obs()

    _, waited, settled = ryp.settled_obs(Shaking(), target + 1.0, running)
    assert not settled and waited >= 0.15


def test_check_robot_metadata():
    ok = {"action_type": "joint", "action_shape": [14], "control_hz": 30, "max_joint_delta": 0.35}
    assert ryp.check_robot_metadata(ok, 30.0) == []
    assert ryp.check_robot_metadata({"action_type": "joint", "action_shape": [14]}, 30.0) == []
    assert "10" in ryp.check_robot_metadata(dict(ok, control_hz=10), 30.0)[0]


def test_policy_error_replies_raise():
    class FakeWS:
        def __init__(self, reply):
            self.reply = reply

        def send(self, _):
            pass

        def recv(self, timeout=None):
            return json.dumps(self.reply)

    conn = ryp.PolicyConnection.__new__(ryp.PolicyConnection)
    conn.timeout = 1.0
    conn._ws = FakeWS({"type": "error", "code": "unknown_message_type", "message": "Unknown message type: chunk"})
    with pytest.raises(ryp.PolicyServerError, match="does not handle chunk requests"):
        conn.predict_chunk({}, HOME, "p")
    conn._ws = FakeWS({"type": "error", "code": "internal_error", "message": "boom"})
    with pytest.raises(ryp.PolicyServerError, match="internal_error: boom"):
        conn.predict_chunk({}, HOME, "p")
