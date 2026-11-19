#!/usr/bin/env bash
# Run the OpenWAM policy server and the YAM client on this machine, next to rd serve.
#
# Until now the policy server (scripts/deploy.sh) and the client
# (scripts/run_yam_policy.py) ran on a GPU host (lambda-server2) and reached rd serve
# on the robot host over the network (--robot-uri ws://mlcav.seas.ucla.edu:8765).
# This script runs all three on the robot host (the Dell, cav-Precision-7865-Tower)
# and connects them over loopback. The policy server binds 127.0.0.1 by default: it
# has no authentication, and this host has a public SEASnet address.
#
#   terminal 1   rd serve --action-type joint --control-hz 30 --max-joint-delta 0.35 \
#                    --resize-images 480x640 --no-depth                 (raiden venv, as before)
#   once         bash scripts/deploy_local.sh server <ckpt_dir> --ckpt-name <file>
#   per rollout  bash scripts/deploy_local.sh run --prompt "Hang the ring on the lower right peg." \
#                    --hold-arm left --video --obs-hz 30 --settle --live-video-delay 4
#   when done    bash scripts/deploy_local.sh stop
#
# Subcommands:
#   server <ckpt_dir> [deploy.py flags]  Start the policy server (openwam conda env) in a
#                 detached tmux session, policy_server_<PORT>, log under outputs/logs/, and
#                 return once it answers a ping (loading takes a few minutes). It refuses a
#                 GPU with less than MIN_FREE_GB free, so it cannot squeeze in beside a
#                 training run. Any deploy.py flag passes through (--ckpt-name,
#                 --denoise-steps, --compile-enabled false, ...).
#   run [run_yam_policy.py flags]  One rollout in the foreground (raiden venv), started
#                 from the repo root so results land in outputs/eval_out/. --policy-uri and
#                 --robot-uri point at loopback; passing either overrides it. The model
#                 is compiled on the client's warm-up chunk, before it asks you to stage the
#                 scene, so the first rollout after `server` waits a few minutes there.
#   status        Server sessions, the checkpoint being served, GPU memory, rd serve.
#   logs          Follow the server log.
#   stop          Ctrl-C the server, then remove its session.
#
# Environment overrides:
#   PORT           policy server port (default 8848). One session per port, so
#                  GPU=1 PORT=8849 serves a second checkpoint beside the first.
#   GPU            GPU index for `server` (default: the one with the most free memory)
#   MIN_FREE_GB    free memory `server` requires on that GPU (default 30; a merged LoRA
#                  checkpoint peaked at 25.5 GiB on an A6000 with --video decoding)
#   HOST           `server` bind address (default 127.0.0.1; 0.0.0.0 serves other hosts too)
#   ROBOT_URI      rd serve's URI for `run` (default ws://127.0.0.1:8765)
#   READY_TIMEOUT  seconds `server` waits for the model to load (default 900)
#   CONDA_ROOT     (default ~/miniconda3)   CONDA_ENV (default openwam)
#   RAIDEN_PY      raiden's venv python (default /data2/takuya/raiden/.venv/bin/python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CALLER_PWD="$PWD"
cd "$REPO_ROOT"

PORT="${PORT:-8848}"
HOST="${HOST:-127.0.0.1}"
ROBOT_URI="${ROBOT_URI:-ws://127.0.0.1:8765}"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-openwam}"
RAIDEN_PY="${RAIDEN_PY:-/data2/takuya/raiden/.venv/bin/python}"

SESSION="policy_server_${PORT}"
POLICY_URI="ws://127.0.0.1:${PORT}"
LOG_DIR="$REPO_ROOT/outputs/logs"
RD_SERVE_HINT="rd serve --action-type joint --control-hz 30 --max-joint-delta 0.35 --resize-images 480x640 --no-depth"

die() {
    echo "[deploy_local] $*" >&2
    exit 1
}
say() { echo "[deploy_local] $*"; }

session_alive() { tmux has-session -t "=$SESSION" 2>/dev/null; }
# outputs/logs/policy_server_<PORT>.latest.log links to the newest server log on that port.
latest_log_link() { echo "$LOG_DIR/policy_server_${1:-$PORT}.latest.log"; }
session_log() { readlink -f "$(latest_log_link "${1:-$PORT}")" 2>/dev/null || true; }
listening() { [[ -n "$(ss -ltnH "sport = :$1" 2>/dev/null)" ]]; }

# Print the policy server's pong as JSON; fail if nothing answers within $2 seconds.
ping_policy() {
    "$RAIDEN_PY" -I - "$1" "${2:-10}" <<'PY'
import json
import sys

from websockets.sync.client import connect

uri, timeout = sys.argv[1], float(sys.argv[2])
try:
    with connect(uri, open_timeout=3, close_timeout=1, max_size=None) as ws:
        ws.send(json.dumps({"type": "ping"}))
        reply = json.loads(ws.recv(timeout=timeout))
except Exception as exc:
    sys.exit(f"{type(exc).__name__}: {exc}")
if reply.get("type") != "pong":
    sys.exit(f"unexpected reply: {reply}")
reply.pop("type")
print(json.dumps(reply))
PY
}

# One line per pong field worth reading: the served weights file and the action representation.
describe_pong() {
    "$RAIDEN_PY" -I -c '
import json, sys
p = json.loads(sys.argv[1])
for key in ("checkpoint", "representation"):
    print(f"  {key:<16}{p.get(key, chr(63))}")' "$1"
}

gpu_report() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader |
        sed 's/^/  GPU /'
}

# ── server ───────────────────────────────────────────────────────────────────
cmd_server() {
    [[ $# -ge 1 && "$1" != -* ]] || die "usage: $0 server <ckpt_dir> [deploy.py flags, e.g. --ckpt-name <file>]"
    local ckpt_dir
    ckpt_dir="$(cd "$CALLER_PWD" && realpath "$1")"
    shift
    [[ -f "$ckpt_dir/config.yaml" ]] || die "$ckpt_dir has no config.yaml; pass a checkpoint run directory"
    for arg in "$@"; do
        case "$arg" in
            --device | --device=* | --host | --host=* | --port | --port=*)
                die "set GPU / HOST / PORT in the environment instead of $arg" ;;
        esac
    done
    session_alive && die "a server is already running on port $PORT (session $SESSION); stop it, or use another PORT"
    listening "$PORT" && die "port $PORT is already in use by another process"
    [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || die "no conda at $CONDA_ROOT (set CONDA_ROOT)"

    if [[ -z "${GPU:-}" ]]; then
        GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
            sort -t, -k2,2nr -k1,1n | head -n1 | cut -d, -f1)"
    fi
    local free_mib
    free_mib="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)" ||
        die "no GPU $GPU"
    if ((free_mib < MIN_FREE_GB * 1024)); then
        echo "[deploy_local] GPU $GPU has $((free_mib / 1024)) GiB free; the server needs ${MIN_FREE_GB} GiB (MIN_FREE_GB)." >&2
        gpu_report >&2
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader |
            sed 's/^/  /' >&2
        exit 1
    fi

    mkdir -p "$LOG_DIR"
    local log="$LOG_DIR/${SESSION}_$(date +%Y-%m-%d_%H-%M-%S).log"
    # The session's shell gets the tmux server's environment, not this one: pass settings in.
    local cmd
    cmd="$(printf '%q ' env "PORT=$PORT" "HOST=$HOST" "CONDA_ROOT=$CONDA_ROOT" "CONDA_ENV=$CONDA_ENV" \
        bash "$SCRIPT_DIR/deploy_local.sh" _serve "$GPU" "$log" "$ckpt_dir" "$@")"
    : >"$log"
    ln -sfn "$(basename "$log")" "$(latest_log_link)"
    tmux new-session -d -s "$SESSION" "$cmd"
    say "policy server starting: GPU $GPU, ws://$HOST:$PORT, tmux session $SESSION"
    say "log: $log"

    local start=$SECONDS next_note=$((SECONDS + 30)) pong
    while ((SECONDS - start < READY_TIMEOUT)); do
        if ! session_alive; then
            echo "[deploy_local] the server exited while starting; end of $log:" >&2
            tail -n 40 "$log" >&2
            exit 1
        fi
        if pong="$(ping_policy "$POLICY_URI" 10 2>/dev/null)"; then
            say "ready after $((SECONDS - start)) s"
            describe_pong "$pong"
            say "next: $0 run --prompt \"...\" [run_yam_policy.py flags]"
            return 0
        fi
        if ((SECONDS >= next_note)); then
            say "loading ($((SECONDS - start)) s): $(tail -n 1 "$log" 2>/dev/null | cut -c1-150)"
            next_note=$((SECONDS + 30))
        fi
        sleep 3
    done
    die "no answer after ${READY_TIMEOUT} s; still running in tmux session $SESSION (see $log)"
}

# Runs inside the tmux session: the server itself, with its console teed to the log.
cmd__serve() {
    local gpu="$1" log="$2" ckpt_dir="$3"
    shift 3
    set +u # conda's activate scripts read unset variables
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
    # One visible GPU, numbered as nvidia-smi numbers it.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
    local args=(--ckpt-dir "$ckpt_dir" --device cuda:0 --host "$HOST" --port "$PORT" "$@")
    {
        echo "[deploy_local] $(date '+%F %T') $(hostname) GPU $gpu, python $(command -v python)"
        echo "[deploy_local] scripts/deploy.py ${args[*]}"
    } | tee -a "$log"
    local status=0
    # A Ctrl-C (stop) reaches the whole pane. python exits on it; tee -i outlives it to log
    # python's last words, and the no-op trap (not '', which python would inherit as
    # "ignore SIGINT") keeps this shell alive to write the exit line.
    trap ':' INT
    python scripts/deploy.py "${args[@]}" 2>&1 | tee -i -a "$log" || status=${PIPESTATUS[0]}
    echo "[deploy_local] $(date '+%F %T') server exited with status $status" | tee -a "$log"
}

# ── run ──────────────────────────────────────────────────────────────────────
cmd_run() {
    # argparse keeps the last occurrence, so URIs given here override the loopback defaults.
    local policy_uri="$POLICY_URI" robot_uri="$ROBOT_URI" prev=""
    for arg in "$@"; do
        case "$prev" in
            --policy-uri) policy_uri="$arg" ;;
            --robot-uri) robot_uri="$arg" ;;
        esac
        case "$arg" in
            --policy-uri=*) policy_uri="${arg#*=}" ;;
            --robot-uri=*) robot_uri="${arg#*=}" ;;
        esac
        prev="$arg"
    done
    [[ -x "$RAIDEN_PY" ]] || die "no python at $RAIDEN_PY (set RAIDEN_PY to raiden's venv)"

    local pong
    if ! pong="$(ping_policy "$policy_uri" 20 2>&1)"; then
        echo "[deploy_local] no policy server answers at $policy_uri ($pong)" >&2
        die "start one with: $0 server <ckpt_dir> --ckpt-name <file>   (or check: $0 status)"
    fi

    # Only look for rd serve's listening socket. Never connect to it: rd serve stops and
    # homes the arms when its client disconnects.
    local hostport="${robot_uri#*://}"
    hostport="${hostport%%/*}"
    local rhost="${hostport%:*}" rport="${hostport##*:}"
    case "$rhost" in
        127.0.0.1 | localhost | "$(hostname)" | "$(hostname -f 2>/dev/null)" | mlcav.seas.ucla.edu)
            listening "$rport" || die "nothing listens on port $rport for $robot_uri; start rd serve first: $RD_SERVE_HINT" ;;
        *) say "rd serve at $robot_uri is not on this host; not checked" ;;
    esac

    exec "$RAIDEN_PY" scripts/run_yam_policy.py --policy-uri "$POLICY_URI" --robot-uri "$ROBOT_URI" "$@"
}

# ── status / logs / stop ─────────────────────────────────────────────────────
cmd_status() {
    local sessions
    sessions="$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^policy_server_' || true)"
    if [[ -z "$sessions" ]]; then
        say "no policy server sessions"
    fi
    local s port log pong
    for s in $sessions; do
        port="${s#policy_server_}"
        log="$(session_log "$port")"
        echo "$s  (ws://127.0.0.1:$port, log $log)"
        if pong="$(ping_policy "ws://127.0.0.1:$port" 5 2>&1)"; then
            describe_pong "$pong"
        elif listening "$port"; then
            echo "  listening, but no pong within 5 s: busy generating a chunk? ($pong)"
        else
            echo "  not listening yet (loading?): $(tail -n 1 "$log" 2>/dev/null | cut -c1-150)"
        fi
    done
    echo "GPUs:"
    gpu_report
    local rport="${ROBOT_URI##*:}"
    rport="${rport%%/*}"
    if listening "$rport"; then echo "rd serve: listening on $rport"; else echo "rd serve: nothing on port $rport"; fi
}

cmd_logs() {
    local log
    log="$(session_log)"
    [[ -f "$log" ]] || die "no server log for port $PORT under $LOG_DIR"
    if session_alive; then
        exec tail -n 100 -f "$log"
    fi
    tail -n 100 "$log"
    say "no server on port $PORT now; that was the end of $log"
}

cmd_stop() {
    if ! session_alive; then
        say "no server on port $PORT (session $SESSION)"
        return 0
    fi
    local log
    log="$(session_log)"
    tmux send-keys -t "=$SESSION:" C-c
    local deadline=$((SECONDS + 30))
    while session_alive && ((SECONDS < deadline)); do sleep 1; done
    if session_alive; then
        say "still running after Ctrl-C; killing session $SESSION"
        tmux kill-session -t "=$SESSION"
        sleep 2
    fi
    local left
    left="$(pgrep -f "scripts/deploy.py .*--port $PORT( |$)" || true)"
    if [[ -n "$left" ]]; then
        say "killing leftover server processes: $left"
        # shellcheck disable=SC2086
        kill $left 2>/dev/null || true
    fi
    say "stopped (log: $log)"
}

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

sub="${1:-}"
[[ $# -gt 0 ]] && shift
case "$sub" in
    server) cmd_server "$@" ;;
    _serve) cmd__serve "$@" ;;
    run) cmd_run "$@" ;;
    status) cmd_status ;;
    logs) cmd_logs ;;
    stop) cmd_stop ;;
    -h | --help | help | "") usage ;;
    *) die "unknown subcommand '$sub' (server | run | status | logs | stop)" ;;
esac
