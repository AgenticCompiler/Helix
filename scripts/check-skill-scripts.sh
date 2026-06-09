#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

changed_skill_scripts=()
while IFS= read -r script_path; do
  changed_skill_scripts+=("$script_path")
done < <(git diff --name-only origin/main...HEAD -- 'skills/*/scripts/*.py')

if [[ "${#changed_skill_scripts[@]}" -eq 0 ]]; then
  echo "No changed skill scripts detected."
  exit 0
fi

for script_path in "${changed_skill_scripts[@]}"; do
  bash scripts/run-skill-script-pyright.sh "$script_path"
done
