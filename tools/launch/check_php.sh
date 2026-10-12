#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source tools/launch/ort_env.sh
action_go_configure_ort_env "$PWD" "check_php"
PHP_NET="${ACTION_GO_NET:-enP8p1s0}"
PHP_BUILD_DIR="${PHP_BUILD_DIR:-build}"
PHP_LOG_DIR="${PHP_LOG_DIR:-outputs/php_hardware_check}"
mkdir -p "$PHP_LOG_DIR"
"$PHP_BUILD_DIR/policy_spec_schema_check" --spec actions/php/config/generic_php.yaml
"$PHP_BUILD_DIR/generic_sensor_policy_player" --spec actions/php/config/generic_php.yaml --check-only
"$PHP_BUILD_DIR/generic_sensor_policy_player" --spec actions/php/config/generic_php.yaml --depth-camera-check-only
"$PHP_BUILD_DIR/generic_sensor_policy_player" --spec actions/php/config/generic_php.yaml \
  --net "$PHP_NET" --dds-dry-run --duration-sec 10 \
  --command-source fixed --command-index 0 \
  --trace-file "$PHP_LOG_DIR/inference.yaml" --event-log "$PHP_LOG_DIR/runtime.jsonl"
