#!/usr/bin/env bash
# Shared ONNX Runtime library-path handling for action_go launchers.
#
# Default mode is ACTION_GO_ORT_MODE=clean:
#   remove ONNX Runtime entries from LD_LIBRARY_PATH and let each player use its
#   own build RPATH/RUNPATH. This supports split builds, e.g.
#   generic_policy_player linked to bundled CPU ORT while generic_composite_policy_player is
#   linked to TRT ORT.
#
# Optional mode ACTION_GO_ORT_MODE=prepend:
#   prepend one selected ORT directory globally. This is useful only when all
#   ONNX players were built against the same ORT.
#
# Set ACTION_GO_SKIP_ORT_ENV=1 or ACTION_GO_ORT_MODE=skip to disable this helper.

_action_go_ort_has_lib() {
  local dir="$1"
  [[ -n "${dir}" && -d "${dir}/lib" ]] || return 1
  [[ -e "${dir}/lib/libonnxruntime.so" || -e "${dir}/lib/libonnxruntime.so.1" ]]
}

_action_go_path_is_ort_lib_dir() {
  local dir="$1"
  [[ -n "${dir}" ]] || return 1
  [[ "${dir}" == *onnxruntime* ]] && return 0
  [[ -e "${dir}/libonnxruntime.so" || -e "${dir}/libonnxruntime.so.1" ]]
}

action_go_clean_library_path() {
  local current="${1:-${LD_LIBRARY_PATH:-}}"
  local item result=""
  [[ -n "${current}" ]] || return 0
  IFS=':' read -r -a _action_go_ld_parts <<< "${current}"
  for item in "${_action_go_ld_parts[@]}"; do
    [[ -z "${item}" ]] && continue
    if _action_go_path_is_ort_lib_dir "${item}"; then
      continue
    fi
    if [[ -z "${result}" ]]; then
      result="${item}"
    else
      result+=":${item}"
    fi
  done
  printf '%s\n' "${result}"
}

action_go_select_ort_dir() {
  local root="$1"
  local arch candidate
  local -a candidates=()

  if [[ -n "${ACTION_GO_ORT:-}" ]]; then
    candidates+=("${ACTION_GO_ORT}")
  fi
  if [[ -n "${ACTION_GO_ONNXRUNTIME_DIR:-}" ]]; then
    candidates+=("${ACTION_GO_ONNXRUNTIME_DIR}")
  fi

  candidates+=(
    "${HOME}/onnxruntime-trt-1.19.2"
    "${HOME}/onnxruntime-trt-1.14.1"
  )

  arch="$(uname -m)"
  case "${arch}" in
    aarch64|arm64)
      candidates+=("${root}/thirdparty/onnxruntime-linux-aarch64-1.22.0")
      ;;
    x86_64|amd64)
      candidates+=("${root}/thirdparty/onnxruntime-linux-x64-1.22.0")
      ;;
  esac

  for candidate in "${candidates[@]}"; do
    if _action_go_ort_has_lib "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

action_go_prepend_library_path() {
  local lib="$1"
  local current
  local item result
  current="$(action_go_clean_library_path "${LD_LIBRARY_PATH:-}")"
  result="${lib}"
  if [[ -n "${current}" ]]; then
    IFS=':' read -r -a _action_go_ld_parts <<< "${current}"
    for item in "${_action_go_ld_parts[@]}"; do
      [[ -z "${item}" || "${item}" == "${lib}" ]] && continue
      result+=":${item}"
    done
  fi
  printf '%s\n' "${result}"
}

action_go_configure_ort_env() {
  local root="$1"
  local label="${2:-action_go}"
  local mode="${ACTION_GO_ORT_MODE:-clean}"
  local ort_dir clean_path

  if [[ "${ACTION_GO_SKIP_ORT_ENV:-0}" == "1" || "${mode}" == "skip" ]]; then
    echo "[${label}] ACTION_GO_ORT_MODE=skip; leaving LD_LIBRARY_PATH unchanged"
    return 0
  fi

  if [[ "${mode}" == "prepend" ]]; then
    if ! ort_dir="$(action_go_select_ort_dir "${root}")"; then
      return 0
    fi
    export ACTION_GO_ORT="${ort_dir}"
    export LD_LIBRARY_PATH="$(action_go_prepend_library_path "${ort_dir}/lib")"
    echo "[${label}] ACTION_GO_ORT_MODE=prepend ACTION_GO_ORT=${ACTION_GO_ORT}"
    return 0
  fi

  clean_path="$(action_go_clean_library_path "${LD_LIBRARY_PATH:-}")"
  if [[ "${clean_path}" != "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${clean_path}"
    echo "[${label}] ACTION_GO_ORT_MODE=clean; removed ONNX Runtime entries from LD_LIBRARY_PATH; using per-player RPATH"
  else
    echo "[${label}] ACTION_GO_ORT_MODE=clean; using per-player RPATH"
  fi
}

action_go_runtime_ld_library_path() {
  local root="$1"
  local mode="${ACTION_GO_ORT_MODE:-clean}"
  local ort_dir
  if [[ "${ACTION_GO_SKIP_ORT_ENV:-0}" == "1" || "${mode}" == "skip" ]]; then
    printf '%s\n' "${LD_LIBRARY_PATH:-}"
    return 0
  fi
  if [[ "${mode}" == "prepend" ]]; then
    if ort_dir="$(action_go_select_ort_dir "${root}")"; then
      ACTION_GO_ORT="${ort_dir}" action_go_prepend_library_path "${ort_dir}/lib"
    else
      action_go_clean_library_path "${LD_LIBRARY_PATH:-}"
    fi
  else
    action_go_clean_library_path "${LD_LIBRARY_PATH:-}"
  fi
}
