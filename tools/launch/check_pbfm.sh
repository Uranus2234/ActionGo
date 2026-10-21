#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/launch/ort_env.sh
action_go_configure_ort_env "$PWD" "check_pbfm"
PBFM_BUILD_DIR="${PBFM_BUILD_DIR:-build}"
PBFM_LOG_DIR="${PBFM_LOG_DIR:-outputs/pbfm_hardware_check}"
PBFM_CHECK_MODE="${1:-offline}"
case "$PBFM_CHECK_MODE" in
  offline|camera|dds) ;;
  *) echo "Usage: $0 [offline|camera|dds]" >&2; exit 2 ;;
esac
for PBFM_SPEC in actions/pbfm/config/generic_pbfm_loco.yaml actions/pbfm/config/generic_pbfm_tracking_dance.yaml actions/pbfm/config/generic_pbfm_pico.yaml; do
  "$PBFM_BUILD_DIR/policy_spec_schema_check" --spec "$PBFM_SPEC"
  "$PBFM_BUILD_DIR/generic_sensor_policy_player" --spec "$PBFM_SPEC" --check-only
done
if [[ "$PBFM_CHECK_MODE" == offline ]]; then exit 0; fi
mkdir -p "$PBFM_LOG_DIR"
PBFM_STATE_ARGS=(--offline-dry-run-state)
if [[ "$PBFM_CHECK_MODE" == dds ]]; then
  PBFM_STATE_ARGS=(--net "${ACTION_GO_NET:-eth0}")
fi
"$PBFM_BUILD_DIR/generic_sensor_policy_player" \
  --spec actions/pbfm/config/generic_pbfm_loco.yaml \
  --dry-run "${PBFM_STATE_ARGS[@]}" --duration-sec 5 \
  --event-log "$PBFM_LOG_DIR/$PBFM_CHECK_MODE.jsonl"
