#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $(basename "$0") <skill-script.py>" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
raw_target="$1"

if [[ "${raw_target}" = /* ]]; then
  target="${raw_target}"
else
  target="${repo_root}/${raw_target}"
fi

if [[ ! -f "${target}" ]]; then
  echo "skill script path does not exist: ${raw_target}" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/triton-agent-uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

target_dir="$(cd "$(dirname "${target}")" && pwd)"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/triton-agent-pyright.XXXXXX")"
trap 'rm -rf "${tmpdir}"' EXIT

cat > "${tmpdir}/pyproject.toml" <<EOF
[tool.pyright]
pythonVersion = "3.11"
typeCheckingMode = "strict"
extraPaths = ["${repo_root}/src", "${target_dir}"]
EOF

cd "${repo_root}"
exec uv run pyright --project "${tmpdir}/pyproject.toml" "${target}"
