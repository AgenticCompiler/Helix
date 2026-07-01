#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md

python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/torch-npu-optimize-knowledge/references/patterns \
  --output skills/triton/torch-npu-optimize-knowledge/references/pattern_index.md

python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-cann-ext-api-patterns/references/patterns \
  --output skills/triton/triton-npu-cann-ext-api-patterns/references/patterns/index.md

python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_symptom_index.py \
  --symptoms-dir skills/triton/triton-npu-optimize-knowledge/references/symptoms \
  --output skills/triton/triton-npu-optimize-knowledge/references/symptom_index.md

echo "Update optimize knowledge indices done."
