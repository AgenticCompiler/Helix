#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: run-gc-pr.sh <create|list|view> [gc-pr-args...]

Runs GitCode pull request commands through uv tool run using the configured wheel URL.
Requires GC_TOKEN to be set.
EOF
}

if [[ "$#" -lt 1 ]]; then
  usage
  exit 2
fi

subcommand="$1"
shift

case "${subcommand}" in
  create|list|view)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "unsupported gc pr subcommand: ${subcommand}" >&2
    usage
    exit 2
    ;;
esac

if [[ -z "${GC_TOKEN:-}" ]]; then
  echo "GC_TOKEN is required to run GitCode PR commands." >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/triton-agent-uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

: "${GITCODE_CLI_WHEEL_URL:=https://gitcode.com/gitcode-cli/cli/releases/download/v0.3.11/gitcode_cli-0.3.11-py3-none-any.whl}"

exec uv tool run --from "${GITCODE_CLI_WHEEL_URL}" gc pr "${subcommand}" "$@"
