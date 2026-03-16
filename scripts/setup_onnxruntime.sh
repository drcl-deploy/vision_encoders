#!/bin/bash
# Downloads and extracts ONNX Runtime GPU to thirdparty/onnxruntime-gpu
# Usage: bash setup_onnxruntime.sh [version]

set -euo pipefail

VERSION="${1:-1.22.0}"
ARCH="x64"
PKG="onnxruntime-linux-${ARCH}-gpu-${VERSION}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PKG}.tgz"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THIRDPARTY_DIR="${SCRIPT_DIR}/../thirdparty"
TARGET_DIR="${THIRDPARTY_DIR}/onnxruntime-gpu"

if [ -d "${TARGET_DIR}" ]; then
    echo "ONNX Runtime GPU already exists at ${TARGET_DIR}"
    echo "Delete it first if you want to re-download."
    exit 0
fi

mkdir -p "${THIRDPARTY_DIR}"
echo "Downloading ONNX Runtime GPU v${VERSION}..."
wget -q --show-progress -O "/tmp/${PKG}.tgz" "${URL}"

echo "Extracting..."
tar -xzf "/tmp/${PKG}.tgz" -C "${THIRDPARTY_DIR}"
mv "${THIRDPARTY_DIR}/${PKG}" "${TARGET_DIR}"
rm -f "/tmp/${PKG}.tgz"

echo "Installed to: ${TARGET_DIR}"
echo "Contents:"
ls "${TARGET_DIR}/lib/" | head -10
