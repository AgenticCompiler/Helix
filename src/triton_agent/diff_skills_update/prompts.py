from __future__ import annotations

import difflib
import json
from pathlib import Path

from triton_agent.diff_skills_update.models import OperatorPair


PATTERN_UPDATE_GUIDANCE = """Pattern update guidance:
- Map changes to pattern cards semantically, not by keyword or cited filename alone.
- Treat logs, summaries, attempts, and citations as evidence hints; confirm the
  actual mechanism from code structure, before/after diffs, correctness, and
  performance/profile outcomes when available.
- If no existing card's `## Summary` and `## Use When` are an honest fit, add a
  new generic pattern card instead of forcing the evidence into a near match.
- Update durable guidance in the main card sections. Prefer refining
  `## Use When`, `## Avoid When`, `## Signals`, and
  `## What To Verify After Applying` over appending round-specific notes.
- Preserve useful existing guidance unless the new evidence clearly supersedes
  it. Integrate successful cases, failures, anti-signals, and stop conditions.
- Keep final card prose kernel-agnostic, self-contained, and free of round IDs or
  artifact-path narration except for concise illustrative examples."""


def build_diff_to_skill_prompt(
    pair: OperatorPair,
    *,
    skills_dir: Path,
    output_json: Path,
) -> str:
    diff_text = _unified_diff(pair.baseline_path, pair.expected_path)
    process_context = _process_context_text(pair)
    return f"""You are updating Triton Ascend NPU optimization knowledge.

Baseline file: {pair.baseline_path}
Optimized answer file: {pair.expected_path}
Editable skills directory: {skills_dir}
Input kind: {pair.source_kind}
{process_context}

Analyze the available optimization evidence. For `optimize-process` inputs,
start from `learned_lessons.md`, then cross-check `opt-note.md`, round summaries,
attempt logs, round states, optional perf/profile analysis, and the before/after
code diff. For plain `diff` inputs, infer the mechanism from the code diff and
any nearby evidence included in the operator directory. Update relevant pattern
cards or add a new generic pattern card when the mechanism is not covered under:
{skills_dir}/triton-npu-optimize-knowledge/references/patterns

{PATTERN_UPDATE_GUIDANCE}

Keep the skill content generic and reusable. Do not copy operator-specific names
unless they are necessary inside a concise example. After editing or adding
pattern cards, regenerate the pattern index if needed.

Pattern card format is mandatory:
- Every pattern card must begin with `# <Human Title>`.
- The first section after the title must be `## Summary`.
- The second section must be `## Use When`.
- Do not put warning, checklist, mandatory, priority, or "check first" sections
  before `## Summary` and `## Use When`.
- Put detection rules under `## Use When` or `## Signals`.
- Put verification rules under `## What To Verify After Applying`.

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

When updating skills, explain the missing guidance in terms of semantic
preconditions, exact code-shape change, observed mismatch, and what should be
verified next. If the candidate failed because the current card overgeneralized,
add an `Avoid When`, anti-signal, or verification rule rather than only adding a
new positive example.

{PATTERN_UPDATE_GUIDANCE}

Pattern card format is mandatory:
- Every pattern card must begin with `# <Human Title>`.
- The first section after the title must be `## Summary`.
- The second section must be `## Use When`.
- Do not put warning, checklist, mandatory, priority, or "check first" sections
  before `## Summary` and `## Use When`.
- Put detection rules under `## Use When` or `## Signals`.
- Put verification rules under `## What To Verify After Applying`.

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


def _process_context_text(pair: OperatorPair) -> str:
    if pair.source_kind != "optimize-process":
        return ""
    lines = ["Optimization process evidence:"]
    if pair.learned_lessons_path is not None:
        lines.append(f"- learned_lessons: {pair.learned_lessons_path}")
    if pair.opt_note_path is not None:
        lines.append(f"- opt_note: {pair.opt_note_path}")
    for path in pair.context_paths:
        if path == pair.learned_lessons_path or path == pair.opt_note_path:
            continue
        lines.append(f"- round_context: {path}")
    return "\n".join(lines)
