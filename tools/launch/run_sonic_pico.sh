#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ACTION_GO_SONIC_ZMQ_ENDPOINT:-tcp://127.0.0.1:5556}"
TOPIC="${ACTION_GO_SONIC_ZMQ_TOPIC:-pose}"
EP="${ACTION_GO_SONIC_EP:-auto}"
CACHE_DIR="${ACTION_GO_SONIC_TRT_CACHE_DIR:-/tmp/action_go_sonic_v1_1_trt_cache}"
INPUT_TIMEOUT_SEC="${ACTION_GO_SONIC_INPUT_TIMEOUT_SEC:-30}"
PICO_AUTOSTART="${ACTION_GO_SONIC_PICO_AUTOSTART:-1}"
PICO_INPUT_SOURCE="${ACTION_GO_SONIC_PICO_INPUT_SOURCE:-xrt}"
PICO_USE_CUDA="${ACTION_GO_SONIC_PICO_CUDA:-0}"
PICO_VIS="${ACTION_GO_SONIC_PICO_VIS:-0}"
PICO_PID=""
PLAYER_PID=""

case "${PICO_AUTOSTART}" in
  0|1) ;;
  *)
    echo "[sonic-pico] ACTION_GO_SONIC_PICO_AUTOSTART must be 0 or 1" >&2
    exit 2
    ;;
esac

case "${PICO_INPUT_SOURCE}" in
  xrt|isaac-teleop) ;;
  *)
    echo "[sonic-pico] ACTION_GO_SONIC_PICO_INPUT_SOURCE must be xrt or isaac-teleop" >&2
    exit 2
    ;;
esac

cleanup() {
  if [[ -n "${PICO_PID}" ]] && kill -0 "${PICO_PID}" 2>/dev/null; then
    echo "[sonic-pico] stopping ActionGo-managed PICO streamer pid=${PICO_PID}"
    kill "${PICO_PID}" 2>/dev/null || true
    wait "${PICO_PID}" 2>/dev/null || true
  fi
}

handle_signal() {
  if [[ -n "${PLAYER_PID}" ]]; then
    kill "${PLAYER_PID}" 2>/dev/null || true
  fi
  exit 143
}

trap cleanup EXIT
trap handle_signal INT TERM HUP

find_gr00t_root() {
  local candidate
  if [[ -n "${ACTION_GO_GR00T_ROOT:-}" ]]; then
    printf '%s\n' "${ACTION_GO_GR00T_ROOT}"
    return
  fi
  for candidate in \
    "${HOME}/resource/GR00T-WholeBodyControl" \
    "${HOME}/resource/GR00t" \
    "${ROOT}/../GR00T-WholeBodyControl"; do
    if [[ -f "${candidate}/gear_sonic/scripts/pico_manager_thread_server.py" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  return 1
}

start_pico_streamer() {
  local gr00t_root pico_script pico_python pico_port
  local -a pico_args

  if pgrep -f 'gear_sonic/scripts/pico_manager_thread_server.py.*--manager' >/dev/null 2>&1; then
    echo "[sonic-pico] using an existing upstream PICO manager"
    return
  fi

  if [[ ! "${ENDPOINT}" =~ ^tcp://(127\.0\.0\.1|localhost|\[::1\]):([0-9]+)$ ]]; then
    echo "[sonic-pico] managed upstream streamer requires a loopback TCP endpoint, got: ${ENDPOINT}" >&2
    exit 2
  fi
  pico_port="${BASH_REMATCH[2]}"

  if ! gr00t_root="$(find_gr00t_root)"; then
    echo "[sonic-pico] GR00T checkout not found; set ACTION_GO_GR00T_ROOT" >&2
    exit 2
  fi
  pico_script="${gr00t_root}/gear_sonic/scripts/pico_manager_thread_server.py"
  pico_python="${ACTION_GO_SONIC_PICO_PYTHON:-${gr00t_root}/.venv_teleop/bin/python}"
  if [[ ! -f "${pico_script}" ]]; then
    echo "[sonic-pico] upstream streamer not found: ${pico_script}" >&2
    exit 2
  fi
  if [[ ! -x "${pico_python}" ]]; then
    echo "[sonic-pico] teleop Python not found: ${pico_python}" >&2
    echo "[sonic-pico] run GR00T install_scripts/install_pico.sh once, or set ACTION_GO_SONIC_PICO_PYTHON" >&2
    exit 2
  fi

  pico_args=(
    "${pico_script}"
    --manager
    --port "${pico_port}"
    --target_fps 50
    --num_frames_to_send 5
    --input-source "${PICO_INPUT_SOURCE}"
  )
  if [[ "${PICO_USE_CUDA}" == "1" ]]; then
    pico_args+=(--cuda)
  fi
  if [[ "${PICO_VIS}" == "1" ]]; then
    pico_args+=(--vis_vr3pt --vis_smpl)
  fi

  echo "[sonic-pico] starting upstream PICO manager from ${gr00t_root} (CPU/headless=$([[ "${PICO_USE_CUDA}" == "0" && "${PICO_VIS}" == "0" ]] && echo yes || echo no))"
  (
    cd "${gr00t_root}"
    exec env PYTHONUNBUFFERED=1 "${pico_python}" "${pico_args[@]}"
  ) &
  PICO_PID=$!
}

if [[ "${PICO_AUTOSTART}" == "1" ]]; then
  start_pico_streamer
else
  echo "[sonic-pico] upstream PICO manager autostart disabled"
fi

"${ROOT}/build/generic_motion_tracking_player" \
  --input pico-zmq \
  --zmq-connect \
  --zmq-endpoint "${ENDPOINT}" \
  --zmq-topic "${TOPIC}" \
  --require-realtime \
  --input-timeout-sec "${INPUT_TIMEOUT_SEC}" \
  --stale-sec 0.1 \
  --ep "${EP}" \
  --trt-cache-dir "${CACHE_DIR}" \
  "$@" &
PLAYER_PID=$!

set +e
wait "${PLAYER_PID}"
status=$?
set -e
PLAYER_PID=""
exit "${status}"
