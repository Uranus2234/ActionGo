#!/usr/bin/env bash
set -euo pipefail

ACTION_GO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VARIANT="${MJLAB_ACTION_VARIANT:-flat}"
case "$VARIANT" in
  flat)
    default_model="$ACTION_GO_ROOT/actions/mjlab/model/reverse_vault_flat_28000.onnx"
    default_spec="$ACTION_GO_ROOT/actions/mjlab/config/generic_reverse_vault_sim.yaml"
    default_max_frames=368
    ;;
  terrain)
    default_model="$ACTION_GO_ROOT/actions/mjlab/model/reverse_vault_terrain_20000.onnx"
    default_spec="$ACTION_GO_ROOT/actions/mjlab/config/generic_reverse_vault_terrain_sim.yaml"
    default_max_frames=368
    ;;
  dance1_subject2)
    default_model="$ACTION_GO_ROOT/actions/mjlab/model/dance1_subject2_29999.onnx"
    default_spec="$ACTION_GO_ROOT/actions/mjlab/config/generic_dance1_subject2_sim.yaml"
    default_max_frames=6574
    ;;
  *)
    echo "MJLAB_ACTION_VARIANT must be flat, terrain, or dance1_subject2; got: $VARIANT" >&2
    exit 2
    ;;
esac
MODEL="${MJLAB_ACTION_MODEL:-$default_model}"
SPEC="${MJLAB_ACTION_SPEC:-$default_spec}"
SIM_BIN="${UNITREE_MUJOCO_BIN:?Set UNITREE_MUJOCO_BIN to the unitree_mujoco executable}"
SCENE="${MJLAB_MUJOCO_SCENE:?Set MJLAB_MUJOCO_SCENE to the matching exported MJB scene}"
INTERFACE="${UNITREE_DDS_INTERFACE:-lo}"
DOMAIN_ID="${UNITREE_DDS_DOMAIN_ID:-0}"
MAX_FRAMES="${MJLAB_ACTION_MAX_FRAMES:-$default_max_frames}"

if [[ ! -x "$SIM_BIN" ]]; then
  echo "unitree_mujoco binary not found: $SIM_BIN" >&2
  exit 1
fi
if [[ ! -x "$ACTION_GO_ROOT/build/generic_policy_player" ]]; then
  echo "generic_policy_player not found; build ActionGo first." >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "ONNX model not found: $MODEL" >&2
  exit 1
fi
if [[ ! -f "$SCENE" ]]; then
  echo "flat MJLab MJB not found: $SCENE" >&2
  exit 1
fi

SIGNAL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/action_go_mjlab.XXXXXX")"
START_FILE="${MJLAB_START_FILE:-$SIGNAL_DIR/start}"
READY_FILE="${MJLAB_READY_FILE:-$SIGNAL_DIR/ready}"
sim_pid=""
player_pid=""
cleanup() {
  set +e
  [[ -n "$player_pid" ]] && kill "$player_pid" 2>/dev/null || true
  [[ -n "$sim_pid" ]] && kill "$sim_pid" 2>/dev/null || true
  [[ -n "$player_pid" ]] && wait "$player_pid" 2>/dev/null || true
  [[ -n "$sim_pid" ]] && wait "$sim_pid" 2>/dev/null || true
  rm -f "$START_FILE" "$READY_FILE"
  rmdir "$SIGNAL_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[mjlab/actiongo] variant=$VARIANT model=$MODEL"
echo "[mjlab/actiongo] starting C++ unitree_mujoco with exact MJB"
"$SIM_BIN" \
  -r g1 -s "$SCENE" -i "$DOMAIN_ID" -n "$INTERFACE" \
  --start-paused --manual-start --start-file "$START_FILE" \
  --lockstep-decimation 4 --lockstep-realtime &
sim_pid=$!

echo "[mjlab/actiongo] press Space in MuJoCo; the C++ player will start immediately afterwards"
while [[ ! -f "$START_FILE" ]]; do
  if ! kill -0 "$sim_pid" 2>/dev/null; then
    echo "unitree_mujoco exited before Space was pressed" >&2
    wait "$sim_pid"
    exit 1
  fi
  sleep 0.05
done

echo "[mjlab/actiongo] starting C++ generic_policy_player"
"$ACTION_GO_ROOT/build/generic_policy_player" \
  --net "$INTERFACE" \
  --domain-id "$DOMAIN_ID" \
  --cmd-topic rt/lowcmd \
  --state-topic rt/lowstate \
  --spec "$SPEC" \
  --model "$MODEL" \
  --max-frames "$MAX_FRAMES" \
  --ready-file "$READY_FILE" \
  --start-file "$START_FILE" \
  --no-damping-on-stop &
player_pid=$!

wait "$sim_pid"
