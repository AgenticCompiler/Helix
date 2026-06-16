from __future__ import annotations

import difflib
import json
from pathlib import Path

from triton_agent.diff_skills_update.models import OperatorPair


def build_diff_to_skill_prompt(
    pair: OperatorPair,
    *,
    skills_dir: Path,
    output_json: Path,
) -> str:
    diff_text = _unified_diff(pair.baseline_path, pair.expected_path)
    return f"""You are updating Triton Ascend NPU optimization knowledge.

Baseline file: {pair.baseline_path}
Optimized answer file: {pair.expected_path}
Editable skills directory: {skills_dir}

Analyze the unified diff below. Update relevant pattern cards or add a new
generic pattern card when the mechanism is not covered under:
{skills_dir}/triton-npu-optimize-knowledge/references/patterns

Keep the skill content generic and reusable. Do not copy operator-specific names
unless they are necessary inside a concise example. After editing or adding
pattern cards, regenerate the pattern index if needed.

Write JSON to {output_json} with this shape:
{{
  "matched_patterns": ["pattern-card-name-or-title"],
  "updated_patterns": ["pattern-card-name-or-title-that-was-added-or-edited"],
  "summary": "short explanation of the mechanism"
}}

Unified diff:
```diff
{diff_text}
```
"""


def build_simulate_prompt(
    *,
    baseline_filename: str,
    candidate_filename: str,
    matched_patterns: list[str],
    output_json: Path,
) -> str:
    patterns_json = json.dumps(matched_patterns, ensure_ascii=True)
    return f"""You are simulating an optimizer worker.

You may read the baseline operator file in the current directory:
{baseline_filename}

You may read the staged skills in this workspace. Do not read parent
directories, do not look for answer files, and do not use any diff report.

Matched patterns to apply:
{patterns_json}

Generate optimized code and write it to:
{candidate_filename}

Also write JSON to {output_json.name} with this shape:
{{
  "summary": "short explanation of the generated code",
  "applied_patterns": ["pattern-card-name-or-title"]
}}
"""


def build_analysis_prompt(
    *,
    pair: OperatorPair,
    candidate_path: Path,
    skills_dir: Path,
    output_json: Path,
) -> str:
    return f"""You are auditing a simulated optimization result.

Baseline file: {pair.baseline_path}
Expected optimized answer file: {pair.expected_path}
Generated candidate file: {candidate_path}
Editable skills directory: {skills_dir}

Compare the generated candidate with the expected optimized answer. Judge
whether the candidate captures the same optimization mechanism and important
code changes. If it does not, update relevant generic skill pattern cards or add
a new generic pattern card in the editable skills directory so the next simulate
iteration has better guidance.

Write JSON to {output_json} with this shape:
{{
  "aligned": true,
  "summary": "short reason",
  "updated_patterns": ["pattern-card-name-or-title-that-was-added-or-edited"],
  "skill_updates": ["changed pattern card or guidance"]
}}
"""


def _unified_diff(before: Path, after: Path) -> str:
    before_lines = before.read_text(encoding="utf-8").splitlines(keepends=True)
    after_lines = after.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=str(before),
            tofile=str(after),
        )
    )
