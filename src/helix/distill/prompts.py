from __future__ import annotations

import difflib
import json
from pathlib import Path

from helix.distill.agent import DISTILL_SKILL_NAME
from helix.distill.knowledge_workspace import (
    editable_knowledge_dir,
    optimize_knowledge_skill_name,
)
from helix.distill.models import OperatorPair


def build_distill_prompt(
    pair: OperatorPair,
    *,
    skills_dir: Path,
    output_json: Path,
    language: str = "triton",
) -> str:
    knowledge_skill = optimize_knowledge_skill_name(language)
    optimize_process_hint = ""
    if pair.source_kind == "optimize-process":
        optimize_process_hint = (
            "\nInput kind is optimize-process. Follow optimize-process steps in "
            "the staged skill's Distillation Workflow section.\n"
        )
    return f"""Use the staged {DISTILL_SKILL_NAME} skill to distill optimization evidence into reusable {language} Ascend NPU pattern knowledge.

Baseline file: {pair.baseline_path}
Optimized answer file: {pair.expected_path}
Operator language: {language}
Editable skills directory: {skills_dir}
Editable knowledge skill: {skills_dir}/{knowledge_skill}
Input kind: {pair.source_kind}
{_process_context_text(pair)}
{optimize_process_hint}
Read the staged skill's Distillation Workflow and Pattern Card Contract sections.
Keep card guidance generic and reusable.
Do not copy operator-specific names unless a concise example needs them. Regenerate
the pattern index after editing or adding pattern cards when the local skill
provides an index builder.

Write JSON to {output_json} with this shape:
{{
  "aligned": true_or_false,
  "matched_patterns": ["pattern-card-name-or-title"],
  "updated_patterns": ["pattern-card-name-or-title-that-was-added-or-edited"],
  "summary": "short explanation of the mechanism"
}}

Unified diff:
```diff
{_unified_diff(pair.baseline_path, pair.expected_path)}
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
    return f"""Use the staged {DISTILL_SKILL_NAME} skill's simulation rules.

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
    return f"""Use the staged {DISTILL_SKILL_NAME} skill's analysis rules.

Baseline file: {pair.baseline_path}
Expected optimized answer file: {pair.expected_path}
Generated candidate file: {candidate_path}
Editable skills directory: {skills_dir}

Compare the generated candidate with the expected optimized answer. Judge whether
the candidate captures the same optimization mechanism and important code
changes. If it does not, update relevant generic pattern cards or add a new
generic pattern card in the editable skills directory so the next simulate
iteration has better guidance. Follow the staged skill's Pattern Card Contract
when editing cards.

Write JSON to {output_json} with this shape:
{{
  "aligned": true,
  "summary": "short reason",
  "updated_patterns": ["pattern-card-name-or-title-that-was-added-or-edited"],
  "skill_updates": ["changed pattern card or guidance"]
}}
"""


def build_post_update_review_prompt(
    *,
    skills_dir: Path,
    updated_patterns: list[str],
    output_json: Path,
    language: str = "triton",
) -> str:
    patterns_list = "\n".join(f"- {name}" for name in sorted(updated_patterns))
    knowledge_dir = editable_knowledge_dir(skills_dir, language=language)
    patterns_dir = knowledge_dir / "references" / "patterns"
    index_path = knowledge_dir / "references" / "pattern_index.md"
    return f"""Use the staged {DISTILL_SKILL_NAME} skill's Post-Update Review section.

Editable skills directory: {skills_dir}
Editable knowledge skill: {knowledge_dir}
Pattern cards directory: {patterns_dir}
Pattern index: {index_path}

Updated pattern cards in this batch:
{patterns_list}

Write JSON to {output_json} using the output contract in the skill's
Post-Update Review section.
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
    if pair.opt_note_path is not None:
        lines.append(f"- opt_note: {pair.opt_note_path}")
    if pair.learned_lessons_path is not None:
        lines.append(f"- learned_lessons: {pair.learned_lessons_path}")
    for path in pair.context_paths:
        if path == pair.learned_lessons_path or path == pair.opt_note_path:
            continue
        lines.append(f"- round_context: {path}")
    return "\n".join(lines)
