#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

exec ./build/generic_policy_player \
  --net lo \
  --cmd-topic rt/lowcmd \
  --state-topic rt/lowstate \
  --spec actions/holosoma/config/generic_holosoma.yaml \
  --model actions/holosoma/model/ppo_g1_29dof_dancing.onnx \
  --hz 50 \
  --publish-hz 200 \
  --max-frames 1020 \
  --no-damping-on-stop \
  "$@"
