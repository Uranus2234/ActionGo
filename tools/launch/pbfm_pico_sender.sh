#!/usr/bin/env bash
set -euo pipefail

PBFM_ACTIONGO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PBFM_SOURCE="${ACTION_GO_PBFM_SOURCE:-${HOME}/algo/fighting/P-BFM}"
PBFM_GMR="${ACTION_GO_GMR_ROOT:-${HOME}/resource/GMR}"
PBFM_PYTHON="${ACTION_GO_PBFM_PICO_PYTHON:-}"

if [[ -z "${PBFM_PYTHON}" ]]; then
  for PBFM_CANDIDATE in "${HOME}/anaconda3/envs/gmr/bin/python" "${HOME}/miniconda3/envs/gmr/bin/python"; do
    if [[ -x "${PBFM_CANDIDATE}" ]]; then
      PBFM_PYTHON="${PBFM_CANDIDATE}"
      break
    fi
  done
fi

if [[ -z "${PBFM_PYTHON}" ]]; then
  echo "[pbfm-pico] set ACTION_GO_PBFM_PICO_PYTHON to a Python with GMR, XRoboToolkit, MuJoCo, scipy and onnxruntime" >&2
  exit 2
fi

exec "${PBFM_PYTHON}" "${PBFM_ACTIONGO_ROOT}/tools/teleop/pbfm_pico.py" \
  --source "${PBFM_SOURCE}" --gmr "${PBFM_GMR}" "$@"
