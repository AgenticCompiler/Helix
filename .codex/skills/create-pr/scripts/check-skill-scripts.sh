#!/usr/bin/env bash
set -euo pipefail

{
  git diff --name-only origin/main -- 'skills/*/scripts/*.py'
  git ls-files --others --exclude-standard -- 'skills/*/scripts/*.py'
} | sort -u | while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  [[ -f "$file" ]] || continue
  bash scripts/run-skill-script-pyright.sh "$file"
done
