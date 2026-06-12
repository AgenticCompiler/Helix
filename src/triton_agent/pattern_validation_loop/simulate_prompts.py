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
            "proposed_code_changes": {
                "summary": "1-3 sentences: what would change in the operator and why.",
                "unified_diff": "Required unified diff vs the current operator file (---/+++ lines, real code).",
                "edits_by_pattern": [
                    {
                        "pattern_id": "pattern-id",
                        "file": "operator.py",
                        "change_type": "tiling | pipeline | memory | launch | other",
                        "before_excerpt": "Minimal quote of code to replace.",
                        "after_excerpt": "Concrete replacement code snippet.",
                        "rationale": "Why this edit implements the pattern on this workspace.",
                    },
                ],
            },
            "proposed_changes": "Same content as proposed_code_changes.summary (legacy string field).",
            "code_plan_quality": "concrete | vague | missing",
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
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    user_prompt: str | None = None,
) -> str:
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
        "- Do not read repo-level performance reports (e.g. `PERF_PATTERN_SYNTHESIS.md`, "
        "`PERF_KNOWLEDGE_BASE.md`), `workspace-plan.json`, or other ground-truth promotion docs.",
        "- Do not read `batch-evaluation.json` at the batch root, any `validation-meta.json`, "
        "`manifest.json`, or `.triton-agent/offline-eval-held/` (ground truth stays outside the operator tree).",
        "",
        "Required (only these inputs):",
        "- Read the workspace operator `.py`, staged `triton-npu-optimize-knowledge` pattern index/cards, "
        "and any `test_*.py.txt` reference files.",
        "- Rank pattern candidates by priority (1 = highest) using **pattern cards and operator code only**. "
        "For each ranked pattern, state whether it **hits** this workspace and give evidence-backed rationale.",
        "- Produce a **concrete code-change plan** as if you were about to edit the operator in a real optimize round:",
        "  - `proposed_code_changes.unified_diff` is **required** (unified diff vs the current operator `.py`).",
        "  - `proposed_code_changes.edits_by_pattern` must list every **hit** pattern with `before_excerpt`, "
        "`after_excerpt`, and Triton/NPU-specific edits (tiling, UB buffering, launch grid, pipelining, etc.).",
        "  - Do not stop at pattern names only; a real worker would leave actionable code behind.",
        "  - Set `code_plan_quality` to `concrete` only when the diff is implementation-ready; use `vague` or "
        "`missing` otherwise.",
        "- Do **not** apply edits to disk; all code stays inside the JSON report.",
        "- Assess whether the **current pattern cards alone** would steer a real optimize agent to produce "
        "this same code plan (`skills_alignment`: aligned | partial | mismatch).",
        f"- Write exactly one JSON file: `{SIMULATE_PLAN_DIR}/{SIMULATE_REPORT_FILENAME}` in this workspace.",
        "",
        "JSON schema (follow field names; `ranked_patterns` must be sorted by ascending `priority`):",
        simulate_report_schema_hint(),
        "",
        f"Operator input: {_display_path(operator_path)}",
        f"Workspace directory: {_display_path(workdir)}",
    ]
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


def build_simulate_skill_audit_prompt(
    *,
    repo_path: Path,
    batch_dir: Path,
    skills_workdir: Path,
    state_path: Path,
    simulate_report_path: Path,
    iteration: int,
    max_iterations: int,
    skill_root: Path,
    knowledge_root: Path,
    record_script: Path,
) -> str:
    return f"""\
Analyze pattern-validation simulate evidence and decide whether the simulate loop can complete.

Read:

  {skill_root.as_posix()}/SKILL.md
  {skill_root.as_posix()}/references/skill-update-contract.md
  {simulate_report_path.as_posix()}
  {knowledge_root.as_posix()}/references/pattern_index.md
  Individual workspace reports under `{batch_dir.as_posix()}/<workspace>/simulate-plan/report.json`

Repository root:

  {repo_path.as_posix()}

Batch root:

  {batch_dir.as_posix()}

Skills workdir (edit pattern cards here only):

  {skills_workdir.as_posix()}

Simulate loop state:

  {state_path.as_posix()}

Current iteration: {iteration} / {max_iterations}

The simulate evidence report `{simulate_report_path.as_posix()}` aggregates per-workspace simulate reports, expected patterns, and heuristic hit results.
`heuristic_suggested_pass` is a **hint only** (whether all expected patterns are hit and simulate status is ok). You must judge whether simulate plans actually applied synthesis-backed mechanisms.

Forbidden (ground truth outside workspaces — do not open):

  {batch_dir.as_posix()}/batch-evaluation.json
  {batch_dir.as_posix()}/workspace-plan.json
  PERF_PATTERN_SYNTHESIS.md / PERF_KNOWLEDGE_BASE.md under the repo root

Required steps:

1. Read `{simulate_report_path.as_posix()}` and per-workspace `simulate-plan/report.json` files only. Do **not** read PERF markdown or `batch-evaluation.json`.
2. For each workspace report, judge simulate pass/fail based on whether the simulate agent's proposed code changes actually implement the target `expected_patterns` and whether the plan is concrete and correct.
   - Do **not** trust simulate self-assessment fields without reading the diff and excerpts.
   - `proposed_code_changes.unified_diff` must be present and implementation-specific when `code_plan_quality` is `concrete`.
   - Every `ranked_patterns[]` entry with `hit: true` should appear in `edits_by_pattern` with matching `before_excerpt`/`after_excerpt`.
   - If the plan is pattern-only prose with no real diff, set `skills_alignment` to `partial` or `mismatch` in your notes and fix cards so the next simulate pass produces code-level guidance.
3. If some workspaces have missing patterns (`heuristic_missing_patterns` not empty) or the code plan is vague/incorrect, update pattern cards under `{knowledge_root.as_posix()}/references/patterns/` to better guide the agent, and regenerate the pattern index:

   python3 {knowledge_root.as_posix()}/scripts/build_pattern_index.py \\
     --patterns-dir {knowledge_root.as_posix()}/references/patterns \\
     --output {knowledge_root.as_posix()}/references/pattern_index.md

4. Record this simulate analyze and skill-audit pass:

   python3 {record_script.as_posix()} \\
     --state {state_path.as_posix()} --phase skill-audit \\
     --note "updated pattern cards from simulate-plan-report"

5. If **every** workspace simulate report passes your independent review (all target `expected_patterns` are hit, structural code plan is concrete and correct, and `skills_alignment: aligned`), mark the simulate loop complete:

   python3 {record_script.as_posix()} \\
     --state {state_path.as_posix()} --phase complete \\
     --note "all workspaces aligned and matched targets after simulate analyze"

   Otherwise stop without `--phase complete` so the CLI runs another simulate iteration with updated skills.

Rules:

- Do not run `optimize-batch` or modify operator workspaces.
- Do not edit staged backend install skills; only `{skills_workdir.as_posix()}`.
- Do not hand-edit `pattern_index.md`.
"""
