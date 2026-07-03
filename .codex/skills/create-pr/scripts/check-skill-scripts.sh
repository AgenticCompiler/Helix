#!/usr/bin/env bash
set -euo pipefail

git diff --name-only origin/main -- 'skills/*/scripts/*.py' | while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  bash scripts/run-skill-script-pyright.sh "$file"
done