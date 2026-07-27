#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVICE_NAME="${ACTION_GO_SERVICE_NAME:-action_go.service}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
RUN_USER="${ACTION_GO_SERVICE_USER:-${SUDO_USER:-$(id -un)}}"
RUN_GROUP="${ACTION_GO_SERVICE_GROUP:-$(id -gn "${RUN_USER}" 2>/dev/null || printf '%s' "${RUN_USER}")}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
if [[ -z "${RUN_HOME}" ]]; then
  RUN_HOME="${ROOT}"
fi
PID_FILE="${ROOT}/logs/action_go_tmux.pid"

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo \
    ACTION_GO_SERVICE_NAME="${SERVICE_NAME}" \
    ACTION_GO_SERVICE_USER="${RUN_USER}" \
    ACTION_GO_SERVICE_GROUP="${RUN_GROUP}" \
    ACTION_GO_NET="${ACTION_GO_NET:-}" \
    "${SCRIPT_PATH}" "$@"
fi

mkdir -p /etc/systemd/system
install -d -o "${RUN_USER}" -g "${RUN_GROUP}" "${ROOT}/logs"

{
  cat <<UNIT
[Unit]
Description=action_go remote supervisor in tmux
Wants=network-online.target
After=network-online.target

[Service]
Type=forking
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${ROOT}
Environment=HOME=${RUN_HOME}
Environment=TERM=xterm-256color
Environment=ACTION_GO_TMUX_SESSION=action_go
Environment=ACTION_GO_TMUX_PID_FILE=${PID_FILE}
PIDFile=${PID_FILE}
UNIT
  if [[ -n "${ACTION_GO_NET:-}" ]]; then
    printf 'Environment=ACTION_GO_NET=%s\n' "${ACTION_GO_NET}"
  fi
  cat <<UNIT
ExecStart=${ROOT}/tools/launch/start.sh start
ExecStop=${ROOT}/tools/launch/start.sh stop
Restart=on-failure
RestartSec=2
TimeoutStartSec=30
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
UNIT
} > "${SERVICE_PATH}"

chmod 0644 "${SERVICE_PATH}"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"

echo "[install_action_go_service] installed ${SERVICE_PATH}"
echo "[install_action_go_service] user=${RUN_USER} group=${RUN_GROUP} home=${RUN_HOME} root=${ROOT}"
if [[ -n "${ACTION_GO_NET:-}" ]]; then
  echo "[install_action_go_service] ACTION_GO_NET=${ACTION_GO_NET} embedded in service"
else
  echo "[install_action_go_service] ACTION_GO_NET not embedded; tools/launch/start_safe.sh default will be used"
fi

echo "[install_action_go_service] service type=forking pid_file=${PID_FILE} restart=on-failure"

if [[ "${1:-}" == "--now" ]]; then
  systemctl restart "${SERVICE_NAME}"
  systemctl status "${SERVICE_NAME}" --no-pager -l || true
else
  echo "[install_action_go_service] start now: sudo systemctl start ${SERVICE_NAME}"
fi
