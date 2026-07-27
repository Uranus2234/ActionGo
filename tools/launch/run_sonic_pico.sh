#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENDPOINT="${ACTION_GO_SONIC_ZMQ_ENDPOINT:-tcp://127.0.0.1:5556}"
TOPIC="${ACTION_GO_SONIC_ZMQ_TOPIC:-pose}"
EP="${ACTION_GO_SONIC_EP:-auto}"
CACHE_DIR="${ACTION_GO_SONIC_TRT_CACHE_DIR:-/tmp/action_go_sonic_trt_cache}"
INPUT_TIMEOUT_SEC="${ACTION_GO_SONIC_INPUT_TIMEOUT_SEC:-30}"

exec "${ROOT}/build/generic_motion_tracking_player" \
  --input realtime-zmq \
  --zmq-connect \
  --zmq-endpoint "${ENDPOINT}" \
  --zmq-topic "${TOPIC}" \
  --require-realtime \
  --input-timeout-sec "${INPUT_TIMEOUT_SEC}" \
  --stale-sec 0.1 \
  --ep "${EP}" \
  --trt-cache-dir "${CACHE_DIR}" \
  "$@"
