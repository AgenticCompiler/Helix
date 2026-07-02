#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

uv run python -m triton_agent.optimize_knowledge.pattern_index \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  --style default

uv run python -m triton_agent.optimize_knowledge.pattern_index \
  --patterns-dir skills/triton/torch-npu-optimize-knowledge/references/patterns \
  --output skills/triton/torch-npu-optimize-knowledge/references/pattern_index.md \
  --style default

uv run python -m triton_agent.optimize_knowledge.pattern_index \
  --patterns-dir skills/triton/triton-npu-cann-ext-api-patterns/references/patterns \
  --output skills/triton/triton-npu-cann-ext-api-patterns/references/patterns/index.md \
  --style default

uv run python -m triton_agent.optimize_knowledge.symptom_index \
  --symptoms-dir skills/triton/triton-npu-optimize-knowledge/references/symptoms \
  --output skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md

echo "Update optimize knowledge indices done."
