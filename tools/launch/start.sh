#!/usr/bin/env bash
set -euo pipefail

# action_go tmux launcher.
# 默认把 tools/launch/start_safe.sh 挂到 tmux 后台，适合 systemd 开机自启动。
# 常用：
#   tools/launch/start.sh           # 后台启动
#   tools/launch/start.sh attach    # 进入 tmux 看实时输出
#   tools/launch/start.sh stop      # Ctrl-C 正常退出，action_go 会切回 WALKRUN

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${ACTION_GO_TMUX_SESSION:-action_go}"
WINDOW="${ACTION_GO_TMUX_WINDOW:-remote}"
LOG_DIR="${ACTION_GO_LOG_DIR:-${ROOT}/logs}"
START_SAFE="${ROOT}/tools/launch/start_safe.sh"
ORT_ENV="${ROOT}/tools/launch/ort_env.sh"
RUNNER="${LOG_DIR}/.action_go_tmux_runner.sh"
PID_FILE="${ACTION_GO_TMUX_PID_FILE:-${LOG_DIR}/action_go_tmux.pid}"
RESTART_DELAY_SEC="${ACTION_GO_RESTART_DELAY_SEC:-2}"

# shellcheck source=/dev/null
source "${ORT_ENV}"

usage() {
  cat <<USAGE
Usage: $0 [start|stop|restart|status|attach|logs]

Environment:
  ACTION_GO_TMUX_SESSION   tmux session name, default: action_go
  ACTION_GO_NET            required DDS network interface
  ACTION_GO_CONFIG         default: config/motion_actions.yaml
  ACTION_GO_REMOTE_MAP     default: config/remote_actions.yaml
  ACTION_GO_ORT_MODE       clean(default): use per-player RPATH; prepend: force one ORT globally
  ACTION_GO_ORT            ORT root used only when ACTION_GO_ORT_MODE=prepend
  ACTION_GO_LOG_DIR        default: ./logs
  ACTION_GO_TMUX_PID_FILE   default: ./logs/action_go_tmux.pid
  ACTION_GO_RESTART_DELAY_SEC restart delay after non-zero exit, default: 2
USAGE
}

need_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[start.sh] ERROR: tmux not found. Install it first: sudo apt install tmux" >&2
    exit 127
  fi
}

has_session() {
  tmux has-session -t "${SESSION}" 2>/dev/null
}

write_pid_file() {
  mkdir -p "$(dirname "${PID_FILE}")"
  tmux display-message -p -t "${SESSION}" "#{pid}" > "${PID_FILE}"
}

runtime_ld_library_path() {
  action_go_runtime_ld_library_path "${ROOT}"
}
write_runner() {
  local log_file latest_log ld_path
  mkdir -p "${LOG_DIR}"
  log_file="${LOG_DIR}/action_go_$(date +%Y%m%d_%H%M%S).log"
  latest_log="${LOG_DIR}/action_go_latest.log"
  ld_path="$(runtime_ld_library_path)"

  ln -sfn "$(basename "${log_file}")" "${latest_log}"

  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${ROOT}"
    printf 'export ACTION_GO_NET=%q\n' "${ACTION_GO_NET:-}"
    printf 'export ACTION_GO_CONFIG=%q\n' "${ACTION_GO_CONFIG:-config/motion_actions.yaml}"
    printf 'export ACTION_GO_REMOTE_MAP=%q\n' "${ACTION_GO_REMOTE_MAP:-config/remote_actions.yaml}"
    printf 'export ACTION_GO_LOG_DIR=%q\n' "${LOG_DIR}"
    printf 'export ACTION_GO_CONSOLE=%q\n' "${ACTION_GO_CONSOLE:-normal}"
    printf 'export LD_LIBRARY_PATH=%q\n' "${ld_path}"
    printf '%s\n' 'export ACTION_GO_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"'
    printf 'LAUNCHER_LOG=%q\n' "${log_file}"
    printf '%s\n' 'launcher_notice() { printf "%s\n" "$*"; printf "%s\n" "$*" >> "${LAUNCHER_LOG}"; }'
    printf 'launcher_notice %q\n' "[start.sh] root=${ROOT} session=${SESSION} log=${log_file}"
    printf '%s\n' 'launcher_notice "[start.sh] ACTION_GO_NET=${ACTION_GO_NET}"'
    printf 'RESTART_DELAY_SEC=%q\n' "${RESTART_DELAY_SEC}"
    printf '%s\n' "trap 'launcher_notice \"[start.sh] runner received stop signal\"; exit 0' INT TERM"
    printf '%s\n' 'while true; do'
    printf '%s\n' '  launcher_notice "[start.sh] launching tools/launch/start_safe.sh at $(date --iso-8601=seconds)"'
    printf '%s\n' '  set +e'
    printf '  python3 tools/logs/manage_logs.py --root %q --prune\n' "${LOG_DIR}"
    printf '  %q\n' "${START_SAFE}"
    printf '%s\n' '  status=$?'
    printf '%s\n' '  set -e'
    printf '%s\n' '  launcher_notice "[start.sh] tools/launch/start_safe.sh exited status=${status} at $(date --iso-8601=seconds)"'
    printf '%s\n' '  if [[ "${status}" -eq 0 || "${status}" -eq 130 || "${status}" -eq 143 ]]; then'
    printf '%s\n' '    launcher_notice "[start.sh] normal exit; tmux session will close"'
    printf '%s\n' '    exit 0'
    printf '%s\n' '  fi'
    printf '%s\n' '  launcher_notice "[start.sh] non-zero exit; restarting in ${RESTART_DELAY_SEC}s"'
    printf '%s\n' '  sleep "${RESTART_DELAY_SEC}"'
    printf '%s\n' 'done'
  } > "${RUNNER}"
  chmod +x "${RUNNER}"

  echo "${log_file}"
}

start_session() {
  need_tmux
  cd "${ROOT}"

  if has_session; then
    write_pid_file || true
    echo "[start.sh] tmux session '${SESSION}' already running."
    echo "[start.sh] attach: tools/launch/start.sh attach"
    return 0
  fi

  : "${ACTION_GO_NET:?Set ACTION_GO_NET to the robot DDS network interface}"

  if [[ ! -x "${START_SAFE}" ]]; then
    echo "[start.sh] ERROR: ${START_SAFE} not executable" >&2
    return 2
  fi
  if [[ ! -x "${ROOT}/build/action_go_remote_test" ]]; then
    echo "[start.sh] ERROR: build/action_go_remote_test not found or not executable." >&2
    echo "[start.sh] Build first: cmake --build build -j" >&2
    return 2
  fi

  local log_file
  log_file="$(write_runner)"
  tmux new-session -d -s "${SESSION}" -n "${WINDOW}" "${RUNNER}"
  sleep 0.3
  if ! has_session; then
    echo "[start.sh] ERROR: tmux session exited immediately; last log lines:" >&2
    tail -80 "${log_file}" >&2 || true
    return 1
  fi
  write_pid_file

  echo "[start.sh] started tmux session '${SESSION}'"
  echo "[start.sh] attach: tools/launch/start.sh attach"
  echo "[start.sh] logs:   tools/launch/start.sh logs"
}

stop_session() {
  need_tmux
  if ! has_session; then
    rm -f "${PID_FILE}"
    echo "[start.sh] tmux session '${SESSION}' is not running."
    return 0
  fi

  echo "[start.sh] stopping '${SESSION}' by Ctrl-C ..."
  tmux send-keys -t "${SESSION}" C-c || true

  for _ in $(seq 1 60); do
    if ! has_session; then
      rm -f "${PID_FILE}"
      echo "[start.sh] stopped."
      return 0
    fi
    sleep 0.2
  done

  echo "[start.sh] session did not exit after Ctrl-C; killing tmux session." >&2
  tmux kill-session -t "${SESSION}" || true
  rm -f "${PID_FILE}"
}

status_session() {
  need_tmux
  if has_session; then
    echo "[start.sh] RUNNING: tmux session '${SESSION}'"
    tmux list-sessions | grep -F "${SESSION}:" || true
    echo "[start.sh] attach: tools/launch/start.sh attach"
  else
    echo "[start.sh] STOPPED: tmux session '${SESSION}' not found"
    return 3
  fi
}

attach_session() {
  need_tmux
  if ! has_session; then
    echo "[start.sh] tmux session '${SESSION}' is not running." >&2
    return 3
  fi
  exec tmux attach-session -t "${SESSION}"
}

follow_logs() {
  local runtime_log="${LOG_DIR}/latest/events.log"
  if [[ -f "${runtime_log}" ]]; then
    exec tail -F "${runtime_log}"
  fi
  local latest_log="${LOG_DIR}/action_go_latest.log"
  if [[ ! -e "${latest_log}" ]]; then
    echo "[start.sh] no log yet: ${latest_log}" >&2
    return 3
  fi
  exec tail -f "${latest_log}"
}

cmd="${1:-start}"
case "${cmd}" in
  start)
    start_session
    ;;
  stop)
    stop_session
    ;;
  restart)
    stop_session || true
    start_session
    ;;
  status)
    status_session
    ;;
  attach)
    attach_session
    ;;
  logs)
    follow_logs
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[start.sh] unknown command: ${cmd}" >&2
    usage >&2
    exit 2
    ;;
esac
