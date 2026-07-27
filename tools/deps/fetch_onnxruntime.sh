#!/usr/bin/env bash
set -euo pipefail

version="1.22.0"
architecture="${1:-$(uname -m)}"
destination="${2:-thirdparty}"

case "$architecture" in
  x86_64|amd64)
    package="onnxruntime-linux-x64-${version}"
    sha256="8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3"
    ;;
  aarch64|arm64)
    package="onnxruntime-linux-aarch64-${version}"
    sha256="bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702"
    ;;
  *)
    echo "Unsupported architecture: $architecture (expected x86_64 or aarch64)" >&2
    exit 2
    ;;
esac

archive="${package}.tgz"
url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive}"
temporary_directory="$(mktemp -d /tmp/actiongo-ort.XXXXXX)"
trap 'rm -rf -- "$temporary_directory"' EXIT

mkdir -p "$destination"
curl --fail --location --retry 3 --output "$temporary_directory/$archive" "$url"
printf '%s  %s\n' "$sha256" "$temporary_directory/$archive" | sha256sum --check --status
tar -xzf "$temporary_directory/$archive" -C "$destination"
echo "Installed $package in $destination/$package"
