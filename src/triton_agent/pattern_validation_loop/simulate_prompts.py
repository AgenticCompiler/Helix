from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from triton_agent.optimize.prompts import (
    cann_ext_api_lines,
    compiler_source_analysis_lines,
    layered_analysis_lines,
    strict_learned_lessons_lines,
)
from triton_agent.pattern_validation_loop.reference_tests import (
    build_pattern_validation_optimize_reference_test_prompt,
)
from triton_agent.prompts import append_additional_user_instructions

SIMULATE_PLAN_DIR = "simulate-plan"
SIMULATE_REPORT_FILENAME = "report.json"
BATCH_SIMULATE_REPORT_FILENAME = "simulate-plan-report.json"


def _display_path(path: Path) -> str:
    return path.as_posix()


def simulate_report_schema_hint() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "workspace": "<workspace_dir_name>",
            "operator_filename": "<operator.py>",
            "expected_patterns": ["pattern-id-from-validation-meta"],
            "ranked_patterns": [
                {
                    "pattern_id": "pattern-id",
                    "priority": 1,
                    "hit": True,
                    "rationale": "Why this pattern matches the operator (code/IR/profile signals).",
                },
            ],
            "patterns_considered_but_rejected": [
                {
                    "pattern_id": "other-pattern-id",
                    "rationale": "Why it does not apply.",
                },
            ],
            "proposed_changes": "Markdown: intended code edits if a real optimize run happened (no file writes).",
            "skills_alignment": "aligned | partial | mismatch",
            "skill_edit_notes": ["Concrete edits to pattern cards under skills workdir if mismatch."],
            "risks_for_real_optimize": ["Compile, correctness, or scope risks for a follow-up optimize-batch."],
        },
        indent=2,
        ensure_ascii=False,
    )


def build_simulate_plan_prompt(
    *,
    operator_path: Path,
    workdir: Path,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str,
    optimize_target: str,
    validation_meta: dict[str, Any] | None,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    user_prompt: str | None = None,
) -> str:
    meta_path = workdir / "validation-meta.json"
    expected_patterns: list[str] = []
    if validation_meta is not None:
        raw = validation_meta.get("expected_patterns", [])
        if isinstance(raw, list):
            expected_patterns = [str(item).strip() for item in raw if str(item).strip()]

    lines = [
        "SIMULATE OPTIMIZE PLAN (dry-run only — highest priority instructions).",
        "You are launched with the same staged optimize skills and workspace layout as a real optimize worker, "
        "but this session is a **simulation** only.",
        "",
        "Forbidden:",
        "- Do not modify the operator file, tests, benches, or deps.",
        "- Do not create or update `baseline/`, `opt-round-*`, `opt-note.md`, or `learned_lessons.md`.",
        "- Do not run pytest, benchmarks, profiling, IR capture, `check-round`, or `check-baseline`.",
        "- Do not run `triton-agent optimize` or `optimize-batch`.",
        "",
        "Required:",
        "- Read the operator, `validation-meta.json`, staged `triton-npu-optimize-knowledge` pattern index/cards, "
        "and any `test_*.py.txt` reference files.",
        "- Rank pattern candidates by priority (1 = highest). For each ranked pattern, state whether it "
        "**hits** this workspace and give evidence-backed rationale.",
        "- Compare your ranking to `expected_patterns` in validation-meta and note alignment or gaps.",
        "- Describe proposed code changes in prose or a unified diff block only inside the report (do not apply).",
        "- Assess whether current pattern cards would steer a real optimize agent correctly (`skills_alignment`).",
        f"- Write exactly one JSON file: `{SIMULATE_PLAN_DIR}/{SIMULATE_REPORT_FILENAME}` in this workspace.",
        "",
        "JSON schema (follow field names; `ranked_patterns` must be sorted by ascending `priority`):",
        simulate_report_schema_hint(),
        "",
        f"Operator input: {_display_path(operator_path)}",
        f"Workspace directory: {_display_path(workdir)}",
        f"Validation meta: {_display_path(meta_path)}",
    ]
    if expected_patterns:
        lines.append(f"Expected patterns from meta: {', '.join(expected_patterns)}")
    else:
        lines.append("Expected patterns from meta: (none listed — infer from synthesis context in meta notes).")
    if test_mode is not None:
        lines.append(f"Real optimize would use test mode: {test_mode}")
    if bench_mode is not None:
        lines.append(f"Real optimize would use bench mode: {bench_mode}")
    lines.append(f"Target chip for a real optimize session: {target_chip}.")
    lines.extend(
        [
            "Use the staged skill `triton-npu-optimize` only as context for how a real worker would think; "
            "do not execute its optimize steps.",
            "Use the staged `triton-npu-optimize-knowledge` skill for pattern and symptom references.",
            "Read `references/pattern_index.md` before opening individual pattern cards.",
            "Inspect the operator file directly when code structure is unclear at pattern triage.",
            *layered_analysis_lines(round_scope="this simulate-plan (analysis only, no round artifacts)"),
            *strict_learned_lessons_lines(),
            "When you finish, ensure the JSON report is valid and saved under "
            f"`{SIMULATE_PLAN_DIR}/{SIMULATE_REPORT_FILENAME}`.",
        ],
    )
    lines.extend(
        compiler_source_analysis_lines(
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        ),
    )
    lines.extend(cann_ext_api_lines(enabled=enable_cann_ext_api))
    if optimize_target == "operator":
        lines.append("Target scope for a real optimize session: operator (end-to-end).")
    else:
        lines.append("Target scope for a real optimize session: kernel.")

    base = "\n".join(lines)
    base = append_additional_user_instructions(
        base,
        build_pattern_validation_optimize_reference_test_prompt(),
    )
    return append_additional_user_instructions(base, user_prompt)
