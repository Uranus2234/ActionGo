#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
export ACTION_GO_NET="${ACTION_GO_NET:-eth0}"
export ACTION_GO_EVENT_LOG="${ACTION_GO_EVENT_LOG:-$PWD/logs/pbfm_supervisor_$(date +%Y%m%d_%H%M%S).jsonl}"
exec bash tools/launch/start_safe.sh "$@"
