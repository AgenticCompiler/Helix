#!/usr/bin/env bash
set -euo pipefail

REPO_URL="git@gitcode.com:ssq0404/BitfunProfilingTool.git"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/skills/common/ascend-npu-run-eval/scripts"

TMP_DIR="$(mktemp -d -t bitfun-profiling-XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "[INFO] Cloning ${REPO_URL} ..."
if ! git clone --depth 1 "${REPO_URL}" "${TMP_DIR}"; then
    echo "[ERROR] git clone failed." >&2
    exit 1
fi

FE_DIR="${TMP_DIR}/feature_extraction"
if [ ! -d "${FE_DIR}" ]; then
    echo "[ERROR] feature_extraction/ not found in cloned repo." >&2
    exit 1
fi

for py_file in "${FE_DIR}"/*.py; do
    [ -f "${py_file}" ] || continue
    cp "${py_file}" "${TARGET_DIR}/"
    echo "[INFO] Copied $(basename "${py_file}")"
done

echo "[INFO] Done. Dependencies installed to ${TARGET_DIR}"
