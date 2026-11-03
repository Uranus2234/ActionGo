#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/tools/launch/ort_env.sh"
action_go_configure_ort_env "${ROOT}" "start_sim"
export ACTION_GO_GIT_SHA="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
exec "${ACTION_GO_SIM_PYTHON:-python3}" "${ROOT}/tools/sim/sim2sim.py" "$@"
