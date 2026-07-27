#!/usr/bin/env bash
set -euo pipefail

# action_go safe launcher.
#
# 所有机器相关的网卡名集中在这里改：
# - 机器人 NX 常见：enP8p1s0
# - 老机器/部分 PC 常见：eth0
# - 你这台机器 ifconfig 里还出现过：wlx782288b51cae / usb0 / usb1 / lo / docker0
#
# 只需要把 ACTION_GO_NET 改成当前机器实际 DDS 网卡名即可。
# 也可以临时覆盖：ACTION_GO_NET=enP8p1s0 tools/launch/start_safe.sh
NET_G1_NX="enP8p1s0"
NET_UNITREE_LAN="$NET_G1_NX"
NET_ETH0="eth0"
NET_USB0="usb0"
NET_USB1="usb1"
NET_WIFI_USB="wlx782288b51cae"
NET_LOOPBACK="lo"
NET_DOCKER0="docker0"

ACTION_GO_NET="${ACTION_GO_NET:-$NET_UNITREE_LAN}"
ACTION_GO_CONFIG="${ACTION_GO_CONFIG:-config/motion_actions.yaml}"
ACTION_GO_REMOTE_MAP="${ACTION_GO_REMOTE_MAP:-config/remote_actions.yaml}"
ACTION_GO_EVENT_LOG="${ACTION_GO_EVENT_LOG:-}"

cd "$(dirname "$0")/../.."

# Prefer the same ONNX Runtime at run time that action_go was built against.
# Override with ACTION_GO_ORT=/path/to/onnxruntime-trt-x.y.z if needed.
# shellcheck source=/dev/null
source tools/launch/ort_env.sh
action_go_configure_ort_env "$PWD" "start_safe"

echo "[start_safe] net=${ACTION_GO_NET} config=${ACTION_GO_CONFIG} remote_map=${ACTION_GO_REMOTE_MAP}"
echo "[start_safe] edit tools/launch/start_safe.sh or set ACTION_GO_NET=... if this machine uses another NIC"

if [[ ! -d "/sys/class/net/${ACTION_GO_NET}" ]]; then
  echo "[start_safe] ERROR: net interface '${ACTION_GO_NET}' not found on this machine." >&2
  echo "[start_safe] Available interfaces:" >&2
  ls /sys/class/net >&2
  echo "[start_safe] Change ACTION_GO_NET near the top of tools/launch/start_safe.sh, or run:" >&2
  echo "  ACTION_GO_NET=<your_nic> tools/launch/start_safe.sh" >&2
  exit 2
fi

EXTRA_ARGS=()
if [[ -n "${ACTION_GO_EVENT_LOG}" ]]; then
  EXTRA_ARGS+=(--event-log "${ACTION_GO_EVENT_LOG}")
fi

exec ./build/action_go_remote_test \
  --net "${ACTION_GO_NET}" \
  --config "${ACTION_GO_CONFIG}" \
  --remote-map "${ACTION_GO_REMOTE_MAP}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
