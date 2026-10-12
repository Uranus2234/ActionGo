#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/launch/ort_env.sh
action_go_configure_ort_env "$PWD" "preview_risk_switch"
MJLAB_ROOT="${MJLAB_ROOT:-$HOME/mjlab}"
SOURCE_ACTION="${1:-paoxiao}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$SOURCE_ACTION" in
  paoxiao|hlqk) SOURCE_ACTION=hlqk ;;
  wbst|*/wbst.onnx) SOURCE_ACTION=wbst ;;
  *) echo "Usage: bash tools/launch/preview_risk_switch.sh [paoxiao|wbst] [sim options]"; exit 2 ;;
esac
TRAINING_ARGS=()
if [[ "$SOURCE_ACTION" == wbst ]]; then
  : "${WBST_TRAINING_ENV:?Set WBST_TRAINING_ENV to the matching training params/env.yaml}"
  TRAINING_ARGS=(--training-env "$WBST_TRAINING_ENV")
fi
exec uv run --project "$MJLAB_ROOT" --extra cu128 python tools/sim/action_go_sim_remote.py \
  --unitree-mujoco-root "$HOME/unitree_mujoco" \
  --unitree-sdk2py-root "$HOME/unitree_sdk2_python" \
  --interface lo --no-use-joystick --safety-disable --auto-action "$SOURCE_ACTION" \
  --reset-auto-action-to-reference \
  --risk-model "${HANDOFFGUARD_RISK_MODEL:?Set HANDOFFGUARD_RISK_MODEL to the preview MLP ONNX}" \
  "${TRAINING_ARGS[@]}" "$@"
