#!/usr/bin/env python3
"""Run an OpenWAM YAM joint-space checkpoint on the real bimanual YAM arms.

Three processes::

    OpenWAM policy server   openwam env, GPU host   bash scripts/deploy.sh <ckpt_dir> --ckpt-name <file>
    raiden rd serve         robot host              rd serve --action-type joint --control-hz 30 ...
    this client             raiden's venv           python scripts/run_yam_policy.py --robot-uri ... --prompt ...

Every chunk, the client sends the newest observation (three camera frames, the
measured 14-D joints and the prompt) as one ``chunk`` request and gets the whole
generated chunk back: absolute joints ``[left q6, left grip, right q6, right grip]``
in physical units, one per 30 fps control step. It clips them to the joint limits
the arms enforce, bounds the per-step change, and dispatches them at
``--control-hz``. The model conditions on that one observation only, so nothing
carries over between chunks; the arms hold their last command while the next
chunk is generated.

The server neither clamps nor rate-limits what it returns, and ``rd serve`` clips
joints silently and e-stops (hold 5 s, home, exit) when a command strays more than
its ``--max-joint-delta`` from the measured joints. The checks here exist so this
client stops first, with the arms holding, and leaves a record of why:

    per chunk   non-finite or misshapen output; first command too far from the
                measured joints (--max-command-gap); joint-limit and gripper
                clipping (counted); per-step slew limit (--max-step-delta), halting
                once it has pulled the trajectory --max-slew-deviation off course
    per step    lag-compensated tracking error (--max-tracking-error) and action
                backlog (--max-backlog), every --check-every steps

One rollout per invocation: disconnecting is what makes ``rd serve`` stop and home
the arms, and it exits afterwards. Startup homes the arms, warms up the policy
server, then waits for Enter while you stage the scene. Ctrl-C once stops
dispatching at the next step with the arms holding; twice exits at once. Either
way these are written to --out, by default
outputs/eval_out/<run>/<checkpoint>/closed_loop/ for the checkpoint the policy
server reports, beside scripts/open_loop_eval.py's open_loop/ for it. After a real
rollout you are asked for a success / fail / exclude label:

    ep<N>_trajectory.png         per joint: predicted, commanded (after clip + slew)
                                 and measured, chunk starts marked (needs matplotlib)
    ep<N>_imagined_vs_live.mp4   with --video: every imagined frame beside the live
                                 canvas --live-video-delay steps after the step it stands
                                 for, and their difference; past --horizon, the chunk's
                                 last live canvas is held
    ep<N>_rollout.npz            every series above, unrounded
    ep<N>_summary.json           settings, halt reason, timing, tracking, clipping, chunk
                                 border steps, and with --video the imagined-vs-live
                                 PSNR per frame offset

Rollouts of a checkpoint are numbered on from the highest ep<N> already in --out,
across sessions: a real-robot rollout cannot be reproduced, so none is ever
overwritten.

Start with ``--dry-run``: the whole protocol against real observations, nothing
sent to the arms. Then ``--step``, which pauses before each chunk.

The arms trail their commands by ~6 steps, and each chunk starts from the measured
state it was given. Observed the moment the previous chunk ends, that state is behind
where the arms come to rest while the next chunk is generated, so the arms step back
at every chunk border. ``--settle`` waits for them to arrive before observing.

This file imports neither torch nor openwam. It needs numpy, Pillow,
websockets>=12 and raiden's chiral, so it runs in raiden's venv and can be copied
to the robot host on its own.
"""

import argparse
import base64
import contextlib
import datetime
import inspect
import io
import json
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- OpenWAM policy-server protocol (client-side mirror of openwam/deploy/server.py) ---
CHUNK = "chunk"
ACTION_CHUNK = "action_chunk"
RESET = "reset"
RESET_ACK = "reset_ack"
PING = "ping"
PONG = "pong"
ERROR = "error"
ERR_UNKNOWN_TYPE = "unknown_message_type"

# --- YAM joint layout, shared by the training data, the policy output and rd serve ---
ARM_DOF = 6
YAM_DOF = ARM_DOF + 1  # 6 arm joints + gripper
BIMANUAL_DOF = 2 * YAM_DOF  # left arm first
ARM_JOINTS = np.r_[0:ARM_DOF, YAM_DOF : YAM_DOF + ARM_DOF]
GRIPPERS = np.array([ARM_DOF, YAM_DOF + ARM_DOF])
ARM_DIMS = {"left": np.arange(0, YAM_DOF), "right": np.arange(YAM_DOF, BIMANUAL_DOF)}
JOINT_NAMES = [f"{side}_{j}" for side in ("left", "right") for j in [*(f"joint_{i}" for i in range(1, 7)), "gripper"]]

# i2rt's YAM limits (i2rt/robots/get_robot.py, _ARM_JOINT_LIMITS[ArmType.YAM]) widened by
# the 0.15 rad buffer get_robot adds before MotorChainRobot, which clips anything outside
# them without saying so. Clipping here first is what makes the clipping visible.
YAM_ARM_JOINT_LIMITS = np.array(
    [[-2.618, 3.054], [0.0, 3.65], [0.0, 3.665], [-1.571, 1.571], [-1.571, 1.571], [-2.094, 2.094]]
) + np.array([-0.15, 0.15])
_ARM_LO = np.tile(YAM_ARM_JOINT_LIMITS[:, 0], 2)
_ARM_HI = np.tile(YAM_ARM_JOINT_LIMITS[:, 1], 2)

# The server places these image keys on the checkpoint's camera_layout
# [cam_high, cam_left_wrist, cam_right_wrist], and raiden's LeRobot export records
# scene_camera as cam_high. --cameras overrides the raiden side.
IMAGE_KEYS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
DEFAULT_CAMERAS = "scene_camera,left_wrist_camera,right_wrist_camera"
PROPRIO_KEYS = ("follower_l_joint_pos", "follower_r_joint_pos")

# How much the error-vs-lag curve must vary before the position of its minimum
# means anything; below this the arm is dwelling and every lag fits equally well.
LAG_IDENTIFIABLE_RAD = 0.05
# Smallest y-span of a trajectory.png panel (rad, or gripper aperture): the floor the
# YAM normalization stats give a joint that never moved.
MIN_PLOT_SPAN = 0.1
# --settle: the arms have arrived within SETTLE_TOL_RAD of their last command on every
# arm joint, or are at rest once none has moved SETTLE_STILL_RAD over SETTLE_STILL_S. The
# second covers an arm resting short of its command under load (~0.02 rad on the YAM);
# SETTLE_STILL_S outlasts the ~0.1 s the command pipeline adds before an arm starts
# moving, so a move that has not begun is not taken for rest. Grippers are left out: one
# closed on the cube never reaches its command.
SETTLE_TOL_RAD = 0.02
SETTLE_STILL_RAD = 0.005
SETTLE_STILL_S = 0.2
SETTLE_TIMEOUT_S = 1.0


class Halt(Exception):
    """A safeguard stopped the rollout. The arms hold their last command."""


class PolicyServerError(RuntimeError):
    """The policy server answered with an error, or not at all."""


class PolicyConnection:
    """One persistent WebSocket to the OpenWAM policy server.

    A chunk request is never retried: a dropped socket mid-rollout means the arms
    stop, not that they wait for a reconnect.
    """

    def __init__(self, url: str, timeout: float):
        from websockets.sync.client import connect

        # ping_interval / proxy only exist on newer sync clients; older ones forward
        # unknown kwargs to socket.create_connection, so pass only what connect() declares.
        kwargs = {"max_size": None, "compression": None, "open_timeout": 10.0, "ping_interval": None, "proxy": None}
        supported = set(inspect.signature(connect).parameters)
        self._ws = connect(url, **{k: v for k, v in kwargs.items() if k in supported})
        self.timeout = timeout

    def _call(self, message: dict, expect: str) -> dict:
        try:
            self._ws.send(json.dumps(message))
            try:
                raw = self._ws.recv(timeout=self.timeout)
            except TypeError:  # websockets < 13: no recv timeout
                raw = self._ws.recv()
        except Exception as exc:
            raise PolicyServerError(f"{message['type']} request failed: {exc!r}") from exc
        reply = json.loads(raw)
        if reply.get("type") == ERROR:
            if reply.get("code") == ERR_UNKNOWN_TYPE and message["type"] == CHUNK:
                raise PolicyServerError(
                    "the policy server does not handle chunk requests; serve from an OpenWAM checkout "
                    "whose openwam/deploy/server.py has CHUNK"
                )
            raise PolicyServerError(f"{reply.get('code')}: {reply.get('message')}")
        if reply.get("type") != expect:
            raise PolicyServerError(f"expected {expect!r} in reply to {message['type']!r}, got {reply.get('type')!r}")
        return reply

    def ping(self) -> dict:
        """The checkpoint's representation contract and the weights file the server loaded."""
        reply = self._call({"type": PING}, PONG)
        return {k: v for k, v in reply.items() if k != "type"}

    def reset(self) -> None:
        self._call({"type": RESET}, RESET_ACK)

    def predict_chunk(
        self, images: Dict[str, str], state: np.ndarray, prompt: str, video: bool = False
    ) -> Tuple[np.ndarray, float, Optional[List[str]]]:
        """``(actions (T, D), server-side ms, imagined frames as base64 PNGs or None)`` for one observation."""
        message = {"type": CHUNK, "images": images, "state": [float(x) for x in state], "prompt": prompt}
        if video:
            message["video"] = True
        reply = self._call(message, ACTION_CHUNK)
        return np.asarray(reply["actions"], dtype=np.float64), float(reply["latency_ms"]), reply.get("video")

    def close(self) -> None:
        self._ws.close()


# --- observations -------------------------------------------------------------


def measured_joints(observation) -> np.ndarray:
    """(14,) measured joints, left arm first: the layout of the training state."""
    try:
        q = np.concatenate([np.asarray(observation.proprios[k], dtype=np.float64).reshape(-1) for k in PROPRIO_KEYS])
    except KeyError as exc:
        raise RuntimeError(f"proprio {exc} missing; is rd serve running with both followers?") from exc
    if q.shape != (BIMANUAL_DOF,):
        raise RuntimeError(f"expected {BIMANUAL_DOF} measured joints, got {q.shape[0]}")
    return q


def camera_frames(observation, cameras: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Image key -> uint8 HxWx3 RGB frame for the three cameras the checkpoint was trained on.

    ``cameras`` maps raiden camera names to image keys. Frames pass through as
    streamed: rd serve's live right wrist frame was checked to be upright, as in
    the training data.
    """
    frames = {}
    for cam in observation.cameras:
        key = cameras.get(cam.name)
        if key is None:
            continue
        image = np.asarray(cam.image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"camera {cam.name}: expected uint8 HxWx3 RGB, got {image.dtype} {image.shape}")
        frames[key] = np.ascontiguousarray(image)
    missing = sorted(set(cameras.values()) - set(frames))
    if missing:
        raise RuntimeError(
            f"no frame for {missing}; rd serve sent {[c.name for c in observation.cameras]}. "
            "Pass --cameras with rd serve's camera names."
        )
    return frames


def encode_png(image: np.ndarray) -> str:
    """Base64 PNG, the lossless encoding the server expects. Level 1 keeps a 480x640 frame to a few ms."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", compress_level=1)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_png(data: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def compose_canvas(frames: Dict[str, np.ndarray], height: int, width: int) -> np.ndarray:
    """The canvas the policy sees: head across the top 2/3, left and right wrist below.

    Mirrors ``assemble_multiview_layout`` (openwam/dataloader/transforms/multiview.py),
    which the server applies to the frames it is sent: each view BILINEAR-stretched
    into its slot.
    """
    from PIL import Image

    top_h = int(round(height * 2.0 / 3.0))
    half_w = width // 2
    slots = (
        ("head_camera", (0, 0), (width, top_h)),
        ("left_wrist_camera", (0, top_h), (half_w, height - top_h)),
        ("right_wrist_camera", (half_w, top_h), (width - half_w, height - top_h)),
    )
    canvas = Image.new("RGB", (width, height))
    for key, corner, size in slots:
        canvas.paste(Image.fromarray(frames[key]).resize(size, Image.BILINEAR), corner)
    return np.asarray(canvas)


# --- command shaping and monitoring ---------------------------------------------


def clip_to_limits(cmds: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Clip arm joints to YAM_ARM_JOINT_LIMITS and grippers to [0, 1]; count what moved."""
    out = np.array(cmds, dtype=np.float64)
    arm = out[:, ARM_JOINTS]
    overshoot = np.maximum(_ARM_LO - arm, arm - _ARM_HI).max(axis=1)
    out[:, ARM_JOINTS] = np.clip(arm, _ARM_LO, _ARM_HI)
    grip = out[:, GRIPPERS]
    out[:, GRIPPERS] = np.clip(grip, 0.0, 1.0)
    return out, {
        "joint_clipped_steps": int((overshoot > 0).sum()),
        "max_joint_overshoot": float(max(0.0, overshoot.max(initial=0.0))),
        "gripper_clipped": int((grip != out[:, GRIPPERS]).sum()),
    }


def slew_limit(cmds: np.ndarray, last_cmd: np.ndarray, max_step: float) -> Tuple[np.ndarray, dict]:
    """Bound each arm joint's change per step to ``max_step``; grippers are left alone.

    Clipping rewrites the trajectory and the policy never learns of it, so
    ``max_deviation`` (how far the limited stream fell behind the requested one)
    is the number to halt on.
    """
    limited = np.array(cmds, dtype=np.float64)
    prev = np.asarray(last_cmd, dtype=np.float64)
    clipped, worst = 0, 0.0
    for i in range(len(limited)):
        delta = limited[i, ARM_JOINTS] - prev[ARM_JOINTS]
        if np.abs(delta).max() > max_step:
            clipped += 1
            worst = max(worst, float(np.abs(delta).max()))
            limited[i, ARM_JOINTS] = prev[ARM_JOINTS] + np.clip(delta, -max_step, max_step)
        prev = limited[i]
    deviation = np.abs(limited[:, ARM_JOINTS] - np.asarray(cmds)[:, ARM_JOINTS]).max(axis=1, initial=0.0)
    return limited, {
        "clipped_steps": clipped,
        "max_requested_step": worst,
        "max_deviation": float(deviation.max(initial=0.0)),
    }


def tracking_error(history: np.ndarray, measured: np.ndarray) -> Tuple[float, Optional[int]]:
    """Smallest arm-joint error between ``measured`` and any recent command, and how many steps back it was.

    The measurement trails the newest command by the arms' own response (~3 steps
    in the demonstrations), rd serve's smoothing and the observation's age, so it
    is matched against every command in ``history`` (oldest first). The lag is
    None when the arm is dwelling and every command matches equally well.
    """
    errs = np.abs(history[:, ARM_JOINTS] - measured[ARM_JOINTS]).max(axis=1)
    best = int(np.argmin(errs))
    lag = len(errs) - 1 - best if errs.max() - errs.min() > LAG_IDENTIFIABLE_RAD else None
    return float(errs[best]), lag


def settled_obs(robot, target: np.ndarray, stopper) -> Tuple[object, float, bool]:
    """The first observation with the arms arrived at, or resting short of, ``target``, their last command.

    The arms trail their commands by ~6 steps and keep moving to the last one while the
    next chunk is generated, so an observation taken as dispatch ends is that far behind
    them when the chunk starts. The chunk starts from that observation, which steps the
    arms back. Returns ``(observation, seconds waited, settled)``; after SETTLE_TIMEOUT_S,
    the latest observation, unsettled.
    """
    t0 = time.perf_counter()
    seen, recent = None, []  # (arrival time, arm joints) of every fresh observation
    while True:
        obs, now = robot.latest_obs, time.perf_counter()
        if obs is not None and obs is not seen:
            seen = obs
            q = measured_joints(obs)[ARM_JOINTS]
            if np.abs(q - target[ARM_JOINTS]).max() <= SETTLE_TOL_RAD:
                return obs, now - t0, True
            recent.append((now, q))
            window = [r for t, r in recent if now - t <= SETTLE_STILL_S]
            if now - recent[0][0] >= SETTLE_STILL_S and max(np.abs(r - q).max() for r in window) <= SETTLE_STILL_RAD:
                return obs, now - t0, True
        if stopper.stop or now - t0 >= SETTLE_TIMEOUT_S:
            return obs, now - t0, False
        time.sleep(0.005)


def check_robot_metadata(meta: dict, control_hz: float) -> List[str]:
    """Mismatches with ``rd serve``'s configuration that nothing would report at run time."""
    fails = []
    shape = list(meta.get("action_shape", []))
    if meta.get("action_type") != "joint" or shape != [BIMANUAL_DOF]:
        fails.append(
            f"rd serve is in {meta.get('action_type')!r} mode with action_shape {shape}; this client sends "
            f"{BIMANUAL_DOF}-D joint commands. Relaunch it with --action-type joint."
        )
    server_hz = meta.get("control_hz")
    if server_hz is None:
        print("  NOTE: rd serve does not report control_hz; make sure it matches --control-hz by hand.")
    elif abs(float(server_hz) - control_hz) > 1e-6:
        # apply_action is fire-and-forget, so a rate mismatch only shows as motion the arms cannot follow.
        fails.append(f"rd serve --control-hz is {server_hz:g} but this client paces at {control_hz:g}.")
    delta = meta.get("max_joint_delta")
    if delta is None:
        print("  NOTE: rd serve does not report max_joint_delta; it was launched with whatever you passed.")
    else:
        print(f"  rd serve e-stops when a command is more than {float(delta):g} rad from the measured joints.")
    return fails


def check_start_state(q: np.ndarray) -> List[str]:
    """Reasons the measured joints cannot be what the checkpoint's state stats describe."""
    problems = []
    grip = q[GRIPPERS]
    if np.any(grip < -0.05) or np.any(grip > 1.05):
        problems.append(
            f"measured grippers {np.round(grip, 3)} are outside [0, 1]; the checkpoint was trained on "
            "i2rt's normalized aperture (0 closed, 1 open)"
        )
    arm = q[ARM_JOINTS]
    outside = (arm < _ARM_LO) | (arm > _ARM_HI)
    if outside.any():
        names = [JOINT_NAMES[i] for i in ARM_JOINTS[outside]]
        problems.append(f"measured {names} outside the YAM joint limits; the proprio layout is not the expected one")
    return problems


# --- operator interaction --------------------------------------------------------


class Stopper:
    """Ctrl-C once to stop dispatching at the next step, with the arms holding; twice to exit at once."""

    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        if self.stop:
            print("\n[stop] second interrupt -- exiting now", flush=True)
            raise KeyboardInterrupt
        self.stop = True
        print("\n[stop] stopping at the next step; the arms hold their last command", flush=True)


@contextlib.contextmanager
def sigint_deferred():
    """Ignore Ctrl-C while the record is written: it is reached after one or two presses already."""
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def wait_for_operator(stopper: Stopper) -> bool:
    """Block until the scene is staged. Runs with the arms homed and the dispatcher not started."""
    if not sys.stdin.isatty():
        print("[pause] stdin is not a tty -- not waiting for the scene")
        return True
    try:
        input("[pause] arms are home. Stage the scene, then press Enter to start (Ctrl-C, Enter to abort): ")
    except EOFError:
        print()
    return not stopper.stop


def ask_outcome() -> Optional[bool]:
    """Operator's verdict once the arms have stopped: True, False, or None for excluded / unanswered."""
    if not sys.stdin.isatty():
        return None
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        while True:
            try:
                answer = input("[outcome] [s]uccess / [f]ail / e[x]clude? ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[outcome] left unlabeled")
                return None
            if answer in ("s", "f", "x"):
                return {"s": True, "f": False, "x": None}[answer]
    finally:
        signal.signal(signal.SIGINT, previous)


# --- the rollout --------------------------------------------------------------------


def new_record() -> dict:
    """Accumulators for one rollout, owned by the caller so an interrupt cannot take them with it."""
    return {
        "halt_reason": None,
        "success": None,
        "start_state": None,
        "chunks": 0,
        "steps": 0,
        "chunk_starts": [],  # control step at which each chunk began
        "chunk_states": [],  # measured joints the chunk was conditioned on
        "predicted": [],  # raw policy output per chunk
        "commanded": [],  # what was dispatched, per step
        "measured": [],  # measured joints at tracking checks
        "measured_steps": [],
        "tracking": [],
        "lags": [],
        "timing_ms": {"encode": [], "server": [], "roundtrip": [], "hold": [], "settle": []},
        "settle_timeouts": 0,
        "clip": {"joint_clipped_steps": 0, "max_joint_overshoot": 0.0, "gripper_clipped": 0},
        "slew_clipped_steps": 0,
        # --video: per chunk, the imagined frames and the live canvas at the step each stands for
        "video": [],
        "video_psnr_db": None,
    }


@dataclass
class ChunkReply:
    actions: np.ndarray  # (T, 14)
    state: np.ndarray  # measured joints the chunk was conditioned on
    frames: Dict[str, np.ndarray]  # camera frames it was conditioned on
    video: Optional[List[str]]  # imagined canvas frames, base64 PNG, with --video
    encode_ms: float
    server_ms: float


def request_chunk(policy: PolicyConnection, observation, args, video: bool = False) -> ChunkReply:
    t0 = time.perf_counter()
    frames = camera_frames(observation, args.camera_map)
    state = measured_joints(observation)
    # zlib releases the GIL: three 480x640 frames take ~17 ms in parallel, ~40 ms in turn.
    with ThreadPoolExecutor(len(frames)) as pool:
        images = dict(zip(frames, pool.map(encode_png, frames.values())))
    encode_ms = (time.perf_counter() - t0) * 1e3
    actions, server_ms, imagined = policy.predict_chunk(images, state, args.prompt, video=video)
    return ChunkReply(actions, state, frames, imagined, encode_ms, server_ms)


def new_video_clip(reply: ChunkReply, start_step: int, delay: int) -> Optional[dict]:
    """Imagined frames of one chunk, with live frame 0: the observation it was conditioned on.

    Imagined frame ``j`` stands for ``j * stride`` executed steps after the chunk's
    observation. It shows the arms as far behind their commands as in the
    demonstrations, and the live arms trail theirs further, so the live canvas for
    it is captured ``delay`` steps later (:func:`capture_live`).
    """
    imagined = reply.video or []
    if len(imagined) < 2 or len(reply.actions) % (len(imagined) - 1):
        print(f"  WARNING: {len(imagined)} imagined frames for a {len(reply.actions)}-step chunk; no video for it")
        return None
    height, width = decode_png(imagined[0]).shape[:2]
    return {
        "start": start_step,
        "stride": len(reply.actions) // (len(imagined) - 1),
        "delay": delay,
        "size": (height, width),
        "imagined": imagined,
        "live": [compose_canvas(reply.frames, height, width)],
    }


def capture_live(robot, clip: dict, step: int, executed: int, args) -> None:
    """Capture the live canvas due at ``step`` of a chunk that executes ``executed`` steps, if one is.

    Imagined frame ``j`` gets the canvas ``clip['delay']`` steps after ``j * stride``.
    A chunk ending off the frame grid (--horizon) also gets one that long after its
    end, held for the frames it did not reach.
    """
    frame_step = step - clip["delay"]
    if not 0 < frame_step <= executed or (frame_step % clip["stride"] and frame_step != executed):
        return
    latest = robot.latest_obs
    canvas = None if latest is None else compose_canvas(camera_frames(latest, args.camera_map), *clip["size"])
    if frame_step % clip["stride"]:
        clip["last"] = (step, canvas)
    else:
        clip["live"].append(canvas)


def shape_chunk(actions: np.ndarray, state: np.ndarray, last_cmd: np.ndarray, hold: Optional[np.ndarray], args, record):
    """Policy output ``(T, 14)`` -> the commands to dispatch, or :class:`Halt`."""
    if not np.isfinite(actions).all():
        raise Halt("policy returned non-finite actions; check the checkpoint's normalization stats")
    if args.horizon is not None and args.horizon > len(actions):
        raise Halt(f"--horizon {args.horizon} exceeds the {len(actions)}-step chunk")
    cmds = actions[: args.horizon].copy()
    if hold is not None:
        cmds[:, ARM_DIMS[args.hold_arm]] = hold
    cmds, clip = clip_to_limits(cmds)
    record["clip"]["joint_clipped_steps"] += clip["joint_clipped_steps"]
    record["clip"]["gripper_clipped"] += clip["gripper_clipped"]
    record["clip"]["max_joint_overshoot"] = max(record["clip"]["max_joint_overshoot"], clip["max_joint_overshoot"])

    # In the demonstrations the command leads the measured joints by at most ~0.22 rad.
    gap = np.abs(cmds[0, ARM_JOINTS] - state[ARM_JOINTS])
    if gap.max() > args.max_command_gap:
        name = JOINT_NAMES[ARM_JOINTS[int(gap.argmax())]]
        raise Halt(
            f"first command is {gap.max():.3f} rad from the measured {name} (> --max-command-gap "
            f"{args.max_command_gap}); the policy is not tracking the robot it sees"
        )

    cmds, slew = slew_limit(cmds, last_cmd, args.max_step_delta)
    record["slew_clipped_steps"] += slew["clipped_steps"]
    if slew["clipped_steps"]:
        print(
            f"  SLEW CLIP: {slew['clipped_steps']}/{len(cmds)} steps limited to {args.max_step_delta} rad "
            f"(largest request {slew['max_requested_step']:.3f}); {slew['max_deviation']:.3f} rad behind the request",
            flush=True,
        )
    if slew["max_deviation"] > args.max_slew_deviation:
        raise Halt(
            f"slew limiter fell {slew['max_deviation']:.3f} rad behind the requested trajectory "
            f"(> --max-slew-deviation {args.max_slew_deviation})"
        )
    return cmds, clip, slew


def run_rollout(robot, policy: PolicyConnection, args, stopper: Stopper, record: dict) -> None:
    """One rollout into the caller-owned ``record``. Safeguard halts land in ``record['halt_reason']``."""
    # chiral never clears its action queue, so a previous run's leftovers would be
    # dispatched straight after homing.
    queue = getattr(robot, "_action_queue", None)
    if queue is not None and len(queue):
        print(f"[reset] discarding {len(queue)} stale queued actions")
        queue.clear()

    # chiral multiplexes one socket without request ids: keep the stream stopped
    # around one-shot calls so a reply cannot be taken for a streamed observation.
    robot.stop_obs_stream()
    observation, _ = robot.reset()
    print(f"[reset] arms homed, measured q14 = {np.round(measured_joints(observation), 3)}")

    # The first request compiles server-side (about a minute cold); pay it while the arms are home.
    t0 = time.perf_counter()
    request_chunk(policy, observation, args, video=args.video)
    print(f"[warmup] first chunk took {time.perf_counter() - t0:.1f}s")

    if not wait_for_operator(stopper):
        record["halt_reason"] = "aborted before the rollout started"
        return

    # Observed after staging, so the first chunk plans against the scene as it is now.
    observation = robot.get_obs()
    q0 = measured_joints(observation)
    record["start_state"] = q0
    print(f"[start] measured q14 = {np.round(q0, 3)}")
    problems = check_start_state(q0)
    if problems:
        raise Halt("; ".join(problems))
    hold = q0[ARM_DIMS[args.hold_arm]] if args.hold_arm else None
    policy.reset()

    robot.start_obs_stream(hz=args.obs_hz)
    # chiral's dispatcher sleeps a fixed interval rather than to a deadline, so at
    # exactly --control-hz it runs ~2% slow and the queue grows. Ticking it faster
    # lets the deadline loop below set the pace.
    robot.start_action_dispatch(hz=args.control_hz * args.dispatch_mult)

    period = 1.0 / args.control_hz
    history: List[np.ndarray] = [q0]
    t_dispatch_end = None
    try:
        for chunk_i in range(args.max_chunks):
            if stopper.stop:
                break
            settle_s = ""
            if chunk_i == 0:
                obs = observation
            elif args.settle:
                obs, waited, arrived = settled_obs(robot, history[-1], stopper)
                record["timing_ms"]["settle"].append(waited * 1e3)
                record["settle_timeouts"] += not arrived
                settle_s = f"settle {waited * 1e3:4.0f}ms{'' if arrived else ' (timed out)'}  "
                if stopper.stop:
                    break
            else:
                obs = robot.latest_obs
            if obs is None:
                raise Halt("no observation from rd serve")
            t_req = time.perf_counter()
            reply = request_chunk(policy, obs, args, video=args.video)
            roundtrip_ms = (time.perf_counter() - t_req) * 1e3
            actions, state, encode_ms, server_ms = reply.actions, reply.state, reply.encode_ms, reply.server_ms
            if actions.ndim != 2 or actions.shape[1] != BIMANUAL_DOF or len(actions) == 0:
                raise Halt(f"policy returned shape {actions.shape}; expected (T, {BIMANUAL_DOF}) joints")
            # Recorded before shaping, so a chunk a safeguard refuses is kept too.
            record["chunk_starts"].append(record["steps"])
            record["chunk_states"].append(state)
            record["predicted"].append(actions)
            record["chunks"] = chunk_i + 1
            clip_video = new_video_clip(reply, record["steps"], args.live_video_delay) if args.video else None
            if clip_video is not None:
                record["video"].append(clip_video)
            timing = record["timing_ms"]
            timing["encode"].append(encode_ms)
            timing["server"].append(server_ms)
            timing["roundtrip"].append(roundtrip_ms)
            # The arms hold the last command during inference, so the next chunk is
            # slew-limited from it -- unless nothing was ever sent and they are still where measured.
            last_cmd = state if args.dry_run else history[-1]
            cmds, clip, slew = shape_chunk(actions, state, last_cmd, hold, args, record)

            if args.step and sys.stdin.isatty():
                input(f"[step] chunk {chunk_i}: {len(cmds)} steps ready. Enter to execute: ")
                if stopper.stop:
                    break
            # Arms stand still from the end of one chunk's dispatch to the start of the next.
            hold_ms = roundtrip_ms if t_dispatch_end is None else (time.perf_counter() - t_dispatch_end) * 1e3
            timing["hold"].append(hold_ms)

            max_track, deadline = 0.0, time.perf_counter()
            for i, cmd in enumerate(cmds):
                if stopper.stop:
                    break
                if not args.dry_run:
                    robot.put_action(cmd.astype(np.float32))
                history.append(cmd)
                record["commanded"].append(cmd)
                record["steps"] += 1
                deadline += period
                time.sleep(max(0.0, deadline - time.perf_counter()))
                if clip_video is not None:
                    capture_live(robot, clip_video, i + 1, len(cmds), args)
                if args.dry_run or (i + 1) % args.check_every:
                    continue
                latest = robot.latest_obs
                if latest is None:
                    continue
                backlog = len(getattr(robot, "_action_queue", ()))
                if backlog > args.max_backlog:
                    raise Halt(
                        f"{backlog} actions queued (> --max-backlog {args.max_backlog}): the dispatcher is "
                        f"{backlog * period:.2f}s behind. Raise --dispatch-mult."
                    )
                measured = measured_joints(latest)
                track, lag = tracking_error(np.asarray(history[-(args.max_obs_lag + 1) :]), measured)
                max_track = max(max_track, track)
                record["measured"].append(measured)
                record["measured_steps"].append(record["steps"])
                record["tracking"].append(track)
                if lag is not None:
                    record["lags"].append(lag)
                if track > args.max_tracking_error:
                    raise Halt(
                        f"tracking error {track:.3f} rad at best lag {lag} (> --max-tracking-error "
                        f"{args.max_tracking_error}); collision or torque limit?"
                    )
            t_dispatch_end = time.perf_counter()
            # The delayed canvases of the chunk's last frames, while the arms close in on its last command.
            if clip_video is not None and not stopper.stop:
                for step in range(len(cmds) + 1, len(cmds) + clip_video["delay"] + 1):
                    deadline += period
                    time.sleep(max(0.0, deadline - time.perf_counter()))
                    capture_live(robot, clip_video, step, len(cmds), args)

            exec_ms = len(cmds) * period * 1e3
            lag_s = f"{np.median(record['lags']):.0f}" if record["lags"] else "-"
            print(
                f"[chunk {chunk_i:3d}] enc {encode_ms:4.0f}ms  server {server_ms:5.0f}ms  "
                f"rtt {roundtrip_ms:5.0f}ms  {settle_s}hold {hold_ms:5.0f}ms  {len(cmds)} steps  "
                f"duty {exec_ms / (exec_ms + hold_ms) * 100:4.1f}%  track {max_track:.3f} rad  lag {lag_s}  "
                f"slew {slew['clipped_steps']}/{len(cmds)}@{slew['max_deviation']:.3f}  "
                f"clip {clip['joint_clipped_steps']}j/{clip['gripper_clipped']}g",
                flush=True,
            )
    except Halt as exc:
        record["halt_reason"] = str(exc)


# --- artifacts --------------------------------------------------------------------


def _percentiles(values) -> Optional[dict]:
    if not len(values):
        return None
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 99)} | {"max": float(np.max(values))}


def chunk_borders(record: dict) -> dict:
    """Per chunk border, worst arm joint (rad): how far the state the chunk was conditioned on
    trailed the last command, and the command step across the border, beside the steps within
    chunks. A chunk planned from a stale state starts behind the arms, so the two border
    figures grow together; --settle exists to shrink them."""
    cmd = np.asarray(record["commanded"], dtype=np.float64).reshape(-1, BIMANUAL_DOF)[:, ARM_JOINTS]
    steps = np.abs(np.diff(cmd, axis=0)).max(axis=1)  # steps[k]: command k -> k + 1
    borders = [(i, s) for i, s in enumerate(record["chunk_starts"]) if 0 < s < len(cmd)]
    trailing = [np.abs(cmd[s - 1] - record["chunk_states"][i][ARM_JOINTS]).max() for i, s in borders]
    border_steps = [steps[s - 1] for _, s in borders]
    within = np.delete(steps, [s - 1 for _, s in borders])
    return {
        "state_behind_last_command": _percentiles(trailing),
        "border_step": _percentiles(border_steps),
        "within_chunk_step": _percentiles(within),
    }


def plot_trajectory(record: dict, path: Path, args) -> bool:
    """Per joint: predicted, commanded and measured against the control step. False without matplotlib."""
    try:
        import matplotlib
    except ImportError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    commanded = np.asarray(record["commanded"], dtype=np.float64).reshape(-1, BIMANUAL_DOF)
    # The state each chunk was conditioned on is a measurement too, and the only one a dry run has.
    measured_steps = [*record["chunk_starts"], *record["measured_steps"]]
    measured = np.asarray([*record["chunk_states"], *record["measured"]], dtype=np.float64).reshape(-1, BIMANUAL_DOF)
    fig, axes = plt.subplots(
        YAM_DOF, 2, figsize=(15, 1.7 * YAM_DOF + 1), sharex=True, squeeze=False, layout="constrained"
    )
    for col, side in enumerate(("left", "right")):
        for row, d in enumerate(ARM_DIMS[side]):
            ax = axes[row, col]
            ax.plot(
                np.arange(len(commanded)), commanded[:, d], color="tab:blue", lw=1.6, label="commanded (clip + slew)"
            )
            # Dashed on top of the solid line, so it shows wherever nothing was clipped, slewed or held.
            for k, (start, predicted) in enumerate(zip(record["chunk_starts"], record["predicted"])):
                p = predicted[: args.horizon, d]
                ax.plot(
                    start + np.arange(len(p)), p, color="tab:orange", lw=1.2, ls="--", label=None if k else "predicted"
                )
            ax.plot(measured_steps, measured[:, d], ls="none", marker="o", ms=2.5, color="black", label="measured")
            for start in record["chunk_starts"][1:]:
                ax.axvline(start, color="0.65", lw=0.8, ls=":", zorder=0)
            # A still arm would otherwise autoscale its sensor noise to full height.
            lo, hi = ax.get_ylim()
            if hi - lo < MIN_PLOT_SPAN:
                mid = (lo + hi) / 2
                ax.set_ylim(mid - MIN_PLOT_SPAN / 2, mid + MIN_PLOT_SPAN / 2)
            ax.set_ylabel(JOINT_NAMES[d], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
        axes[-1, col].set_xlabel(f"control step ({args.control_hz:g} Hz)")
    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    status = " -- dry run" if args.dry_run else ""
    status += " -- HALTED" if record["halt_reason"] else ""
    fig.suptitle(
        f"ep{args.episode}{status}: {record['chunks']} chunks, {record['steps']} steps\n"
        "dotted lines: chunk starts; measured is not lag-compensated (the arm trails its commands by ~5 steps)",
        fontsize=10,
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else min(100.0, 10 * np.log10(255.0**2 / mse))


def video_frames(record: dict) -> Tuple[List[np.ndarray], Dict[int, float]]:
    """imagined | live | |difference| x3 panels for every imagined frame, and mean PSNR per frame offset.

    Imagined frames past the executed steps (--horizon, or a halt) show the chunk's
    last live canvas held, so it sits right before the next chunk's input frame, and
    no difference: they are not counted in the PSNR.
    """
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=13)
    gap, gain = 2, 3
    frames, psnrs = [], {}
    clips = record["video"]
    for ci, clip in enumerate(clips):
        height, width = clip["size"]
        delay = clip.get("delay", 0)
        for j, data in enumerate(clip["imagined"]):
            imagined = decode_png(data)
            executed = j < len(clip["live"])
            if executed:
                live = clip["live"][j]
                live_label = "live" if j == 0 or not delay else f"live, {delay} steps later"
                live_label = live_label if live is not None else "no observation"
            else:
                last_frame = len(clip["live"]) - 1
                last_step = last_frame * clip["stride"] + (delay if last_frame else 0)
                held, live = clip.get("last") or (last_step, clip["live"][-1])
                live_label = f"last live, step {clip['start'] + held}"
            live_panel = live if live is not None else np.zeros_like(imagined)
            if executed and live is not None:
                psnrs.setdefault(j, []).append(_psnr(imagined, live))
                delta = np.abs(imagined.astype(np.int16) - live.astype(np.int16)).mean(axis=2)
                diff_panel = np.repeat(np.clip(delta * gain, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
            else:
                diff_panel = np.zeros_like(imagined)
            out = Image.new("RGB", (3 * width + 2 * gap, height), "white")
            for k, panel in enumerate((imagined, live_panel, diff_panel)):
                out.paste(Image.fromarray(panel), (k * (width + gap), 0))
            draw = ImageDraw.Draw(out)
            label = lambda xy, text: draw.text(xy, text, fill="white", font=font, stroke_width=2, stroke_fill="black")  # noqa: E731
            label((6, 4), "input, VAE reconstruction" if j == 0 else "imagined")
            label((6, 22), f"chunk {ci + 1}/{len(clips)}  frame {j}  step {clip['start'] + j * clip['stride']}")
            label((width + gap + 6, 4), live_label)
            label((2 * (width + gap) + 6, 4), f"|difference| x{gain}" if executed else "not executed")
            frames.append(np.asarray(out))
    return frames, {j: float(np.mean(v)) for j, v in sorted(psnrs.items())}


def write_mp4(frames: List[np.ndarray], path: Path, fps: float) -> str:
    """H.264 through an ffmpeg binary when there is one (plays in browsers and VS Code), else OpenCV's MPEG-4."""
    height, width = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}"]
        cmd += ["-framerate", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"]
        proc = subprocess.Popen([*cmd, "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        try:
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            pass
        proc.stdin.close()
        if proc.wait() == 0:
            return "H.264"
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return "MPEG-4 Part 2 (no ffmpeg found; play it with VLC)"


def episode_file(args, name: str) -> Path:
    """``<out>/ep<N>_<name>``: every file of a rollout carries its episode number."""
    return args.out / f"ep{args.episode}_{name}"


def next_episode(out_dir: Path) -> int:
    """One above the highest ``ep<N>_`` file in ``out_dir``, so numbering continues across sessions.

    Counted from the filenames rather than the summaries: a rollout killed before its
    summary was written still left files, and those must not be overwritten either.
    """
    numbers = [int(m.group(1)) for path in out_dir.glob("ep*_*") if (m := re.match(r"ep(\d+)_", path.name))]
    return max(numbers, default=-1) + 1


def success_tally(out_dir: Path) -> Tuple[int, int]:
    """(succeeded, labelled) over every ``ep<N>_summary.json`` in ``out_dir``; excluded rollouts don't count."""
    labels = []
    for path in out_dir.glob("ep*_summary.json"):
        with contextlib.suppress(OSError, ValueError):
            labels.append(json.loads(path.read_text()).get("success"))
    labelled = [label for label in labels if label is not None]
    return sum(labelled), len(labelled)


def write_media(record: dict, args) -> None:
    """trajectory.png and, with --video, imagined_vs_live.mp4. A failure here never costs the record."""
    try:
        if plot_trajectory(record, episode_file(args, "trajectory.png"), args):
            print(f"  [artifacts] wrote {episode_file(args, 'trajectory.png')}")
        else:
            print("  [artifacts] no matplotlib in this env, so no trajectory.png (rollout.npz has the series)")
    except Exception as exc:
        print(f"  [artifacts] trajectory plot failed: {exc!r}")
    if not record["video"]:
        return
    try:
        frames, record["video_psnr_db"] = video_frames(record)
        fps = args.control_hz / record["video"][0]["stride"]
        codec = write_mp4(frames, episode_file(args, "imagined_vs_live.mp4"), fps)
        print(f"  [artifacts] wrote {episode_file(args, 'imagined_vs_live.mp4')} ({len(frames)} frames, {codec})")
    except Exception as exc:
        print(f"  [artifacts] imagined-vs-live video failed: {exc!r}")


def write_artifacts(record: dict, args, meta: dict) -> dict:
    """rollout.npz (every series) and summary.json (settings, halt reason, statistics)."""
    args.out.mkdir(parents=True, exist_ok=True)
    as_array = lambda key, width: np.asarray(record[key], dtype=np.float64).reshape(-1, *width)  # noqa: E731
    np.savez_compressed(
        episode_file(args, "rollout.npz"),
        predicted=(np.stack(record["predicted"]) if record["predicted"] else np.zeros((0, 0, BIMANUAL_DOF))),
        chunk_starts=np.asarray(record["chunk_starts"], dtype=np.int64),
        chunk_states=as_array("chunk_states", (BIMANUAL_DOF,)),
        commanded=as_array("commanded", (BIMANUAL_DOF,)),
        measured=as_array("measured", (BIMANUAL_DOF,)),
        measured_steps=np.asarray(record["measured_steps"], dtype=np.int64),
        tracking=np.asarray(record["tracking"], dtype=np.float64),
        start_state=np.asarray(record["start_state"] if record["start_state"] is not None else [], dtype=np.float64),
    )
    summary = {
        **meta,
        "success": record["success"],
        "halt_reason": record["halt_reason"],
        "chunks": record["chunks"],
        "steps": record["steps"],
        "start_state": None if record["start_state"] is None else np.round(record["start_state"], 5).tolist(),
        "timing_ms": {k: _percentiles(v) for k, v in record["timing_ms"].items()},
        "tracking_rad": _percentiles(record["tracking"]),
        "lag_steps": _percentiles(record["lags"]),
        "clip": record["clip"],
        "slew_clipped_steps": record["slew_clipped_steps"],
        "chunk_border_rad": chunk_borders(record),
        "settle_timeouts": record["settle_timeouts"],
        # imagined frame offset -> mean PSNR (dB) against the live canvas; 0 is the input's VAE reconstruction
        "video_psnr_db": record["video_psnr_db"],
    }
    episode_file(args, "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --- entry point --------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="Task instruction, verbatim from the dataset's meta/tasks.jsonl.")
    ap.add_argument("--robot-uri", required=True, help="rd serve's chiral URI, e.g. ws://<robot-host>:8765.")
    ap.add_argument("--policy-uri", default="ws://localhost:8848", help="OpenWAM policy server.")
    ap.add_argument("--policy-timeout", type=float, default=300.0, help="Seconds to wait for one chunk.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for the checkpoint's rollouts, each one's files prefixed ep<N>_ (default "
        "outputs/eval_out/<run>/<checkpoint>/closed_loop, for the served checkpoint).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Run everything but never send a command to the arms.")
    ap.add_argument("--step", action="store_true", help="Pause before dispatching each chunk.")
    ap.add_argument(
        "--video",
        action="store_true",
        help="Save imagined_vs_live.mp4. The server decodes every chunk's imagined frames, which lengthens "
        "the pause between chunks; live frames are the latest --obs-hz observation, so raise --obs-hz "
        "(30 if the link allows) for them to land on the step they are paired with.",
    )
    ap.add_argument(
        "--live-video-delay",
        type=int,
        default=4,
        help="With --video, pair each imagined frame with the live canvas this many steps after the step it "
        "stands for. The live arms trail their commands by ~6.8 steps, the demonstrations the imagination "
        "learned from by ~3.3, so at 0 the live panel runs about a frame behind. The last frame's canvas is "
        "taken after the chunk, which without --settle observes the next chunk this much later too.",
    )
    ap.add_argument(
        "--settle",
        action="store_true",
        help="Before observing for the next chunk, wait (at most 1 s) for the arms to reach their last command "
        "or come to rest. They reach it during inference anyway, so a chunk planned from an earlier "
        "observation starts behind them and steps them back. Adds ~0.2 s per chunk.",
    )
    ap.add_argument("--max-chunks", type=int, default=30, help="Chunk budget (~1.07 s of motion each at 30 Hz).")
    ap.add_argument("--horizon", type=int, default=None, help="Steps executed per chunk (default: all of it).")
    ap.add_argument(
        "--hold-arm",
        choices=("left", "right"),
        default=None,
        help="Pin this arm (joints and gripper) to its measured start pose, for a task the other arm does alone.",
    )
    ap.add_argument("--cameras", default=DEFAULT_CAMERAS, help="rd serve's names for the head, left and right camera.")
    ap.add_argument("--control-hz", type=float, default=30.0, help="Must equal rd serve --control-hz and the data fps.")
    ap.add_argument("--obs-hz", type=float, default=10.0, help="Observation stream rate (~2.8 MB per obs at 480x640).")
    ap.add_argument("--dispatch-mult", type=float, default=4.0, help="chiral dispatcher rate as a multiple of control.")
    ap.add_argument(
        "--max-command-gap",
        type=float,
        default=0.25,
        help="Halt if a chunk starts this far (rad) from the measured joints.",
    )
    ap.add_argument("--max-step-delta", type=float, default=0.35, help="Per-step arm-joint change limit, rad.")
    ap.add_argument(
        "--max-slew-deviation",
        type=float,
        default=0.3,
        help="Halt once slew limiting has moved the path this far, rad.",
    )
    ap.add_argument(
        "--max-tracking-error",
        type=float,
        default=0.25,
        help="Halt above this lag-compensated error, rad. Below rd serve's uncompensated --max-joint-delta, "
        "so the client stops first.",
    )
    ap.add_argument("--max-obs-lag", type=int, default=24, help="Recent commands searched by the tracking check.")
    ap.add_argument("--check-every", type=int, default=4, help="Steps between tracking/backlog checks.")
    ap.add_argument("--max-backlog", type=int, default=10, help="Halt if this many actions are queued but unsent.")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.cameras.split(",")]
    if len(names) != 3 or not all(names):
        ap.error("--cameras takes three comma-separated names: head, left wrist, right wrist")
    args.camera_map = dict(zip(names, IMAGE_KEYS))
    if args.horizon is not None and args.horizon <= 0:
        ap.error("--horizon must be positive")
    if args.check_every <= 0:
        ap.error("--check-every must be positive")
    if args.live_video_delay < 0:
        ap.error("--live-video-delay must be non-negative")
    return args


def closed_loop_dir(checkpoint: Optional[str]) -> Path:
    """outputs/eval_out/<run>/<checkpoint>/closed_loop for the weights file the policy server loaded.

    ``<run>`` is the checkpoint's dir under outputs/openwam_checkpoints, e.g.
    place_cube_right_n56/2026-10-23_22-13-11, or its own name when it lies elsewhere.
    scripts/open_loop_eval.py's eval_dir files the same checkpoint's open_loop/ beside this.
    """
    if not checkpoint:
        return Path("outputs/eval_out/unlinked/closed_loop")
    weights = Path(checkpoint).resolve()
    root = Path("outputs/openwam_checkpoints").resolve()
    run = weights.parent.relative_to(root) if weights.parent.is_relative_to(root) else Path(weights.parent.name)
    return Path("outputs/eval_out") / run / weights.stem / "closed_loop"


def main(argv=None) -> int:
    args = parse_args(argv)
    from chiral import PolicyClient  # raiden's venv

    policy = PolicyConnection(args.policy_uri, timeout=args.policy_timeout)
    contract = policy.ping()
    print(f"[policy] {args.policy_uri}: {contract}")
    if contract.get("representation") != "joint":
        raise SystemExit(
            f"the served checkpoint emits {contract.get('representation')!r} actions; this client sends joints"
        )
    if args.out is None:
        if not contract.get("checkpoint"):
            print("WARNING: the policy server does not say which checkpoint it serves; restart it from this checkout")
        args.out = closed_loop_dir(contract.get("checkpoint"))

    args.out.mkdir(parents=True, exist_ok=True)
    args.episode = next_episode(args.out)
    print(f"[out] {args.out}, episode {args.episode}")
    if args.dry_run:
        print("[dry-run] commands will NOT be sent to the arms")
    stopper = Stopper()
    record = new_record()
    meta = {
        "episode": args.episode,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "prompt": args.prompt,
        "policy": {"uri": args.policy_uri, **contract},
        "settings": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("prompt", "episode")
        },
    }
    interrupted = False

    # One connection for the whole rollout: disconnecting e-stops rd serve, which homes the arms and exits.
    with PolicyClient(args.robot_uri) as robot:
        robot_meta = robot.get_metadata()
        print(
            f"[robot] cameras={robot_meta.get('cameras')} control_hz={robot_meta.get('control_hz')} "
            f"max_joint_delta={robot_meta.get('max_joint_delta')}"
        )
        meta["robot"] = {k: robot_meta.get(k) for k in ("action_type", "action_shape", "control_hz", "max_joint_delta")}
        fails = check_robot_metadata(robot_meta, args.control_hz)
        if fails:
            raise SystemExit("\n".join(f"ERROR: {f}" for f in fails))
        try:
            run_rollout(robot, policy, args, stopper, record)
        except PolicyServerError as exc:
            record["halt_reason"] = f"policy server: {exc}"
        except Halt as exc:
            record["halt_reason"] = str(exc)
        except KeyboardInterrupt:
            record["halt_reason"] = "interrupted (second Ctrl-C)"
            interrupted = True
        finally:
            robot.stop_action_dispatch()
            with sigint_deferred():
                write_media(record, args)
                write_artifacts(record, args, meta)

    print(f"[rollout] {record['chunks']} chunks, {record['steps']} steps")
    if record["halt_reason"]:
        print(f"[rollout] HALTED: {record['halt_reason']}")
    clip = record["clip"]
    print(
        f"  clipped: {clip['joint_clipped_steps']} steps to joint limits (worst {clip['max_joint_overshoot']:.3f} rad), "
        f"{clip['gripper_clipped']} gripper values, {record['slew_clipped_steps']} steps by the slew limit"
    )
    if record["tracking"]:
        t = record["tracking"]
        print(f"  tracking error: p50={np.percentile(t, 50):.3f} p99={np.percentile(t, 99):.3f} max={max(t):.3f} rad")
    border = chunk_borders(record)
    if border["border_step"]:
        print(
            f"  chunk borders (p50, worst arm joint): state {border['state_behind_last_command']['p50']:.3f} rad "
            f"behind the last command, step {border['border_step']['p50']:.3f} rad across the border vs "
            f"{border['within_chunk_step']['p50']:.3f} within chunks"
            + (f"; settle timed out {record['settle_timeouts']}x" if record["settle_timeouts"] else "")
        )
    if record["video_psnr_db"]:
        psnr = "  ".join(f"+{j}: {v:.1f}" for j, v in record["video_psnr_db"].items())
        print(f"  imagined vs live PSNR (dB) by frame offset: {psnr}")

    # Asked only after the disconnect has stopped the arms.
    if record["steps"] and not args.dry_run:
        record["success"] = ask_outcome()
        with sigint_deferred():
            write_artifacts(record, args, meta)
        succeeded, labelled = success_tally(args.out)
        verdict = {True: "success", False: "fail", None: "not counted"}[record["success"]]
        print(f"[outcome] ep{args.episode}: {verdict}; {succeeded}/{labelled} labelled episodes here succeeded")
    policy.close()
    print(f"[out] wrote ep{args.episode}_summary.json and ep{args.episode}_rollout.npz to {args.out}")
    return 1 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
