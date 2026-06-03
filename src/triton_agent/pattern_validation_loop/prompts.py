from __future__ import annotations

from pathlib import Path


def build_prepare_prompt(
    *,
    repo_path: Path,
    synthesis_path: Path,
    batch_dir: Path,
    skills_workdir: Path,
    skills_dir: str,
    state_path: Path,
    base_revision: str,
    skill_root: Path,
    knowledge_root: Path,
) -> str:
    iteration_contract = skill_root / "references" / "iteration-contract.md"
    skill_update_contract = skill_root / "references" / "skill-update-contract.md"
    workspace_scaffold_contract = skill_root / "references" / "workspace-scaffold-contract.md"
    scripts_dir = skill_root / "scripts"

    return f"""\
Prepare pattern-validation workspaces for the CLI-driven loop. **Do not** run `optimize-batch` or nested optimize agents.

Read before acting:

  {iteration_contract.as_posix()}
  {skill_update_contract.as_posix()}
  {skill_root.as_posix()}/references/knowledge-base-scaffold-contract.md
  {workspace_scaffold_contract.as_posix()}
  {skill_root.as_posix()}/SKILL.md

Repository root:

  {repo_path.as_posix()}

Synthesis report:

  {synthesis_path.as_posix()}

Knowledge base (read when present at repo root):

  {repo_path.as_posix()}/PERF_KNOWLEDGE_BASE.md

Batch root (create workspaces here):

  {batch_dir.as_posix()}

Persistent loop skills workdir (edit knowledge here only):

  {skills_workdir.as_posix()}

Loop state file:

  {state_path.as_posix()}

Git base revision for pre-optimization snapshots:

  {base_revision}

Required steps:

1. Initialize loop state if `{state_path.as_posix()}` does not exist:

   python3 {scripts_dir.as_posix()}/init_loop_state.py \\
     --repo {repo_path.as_posix()} \\
     --synthesis {synthesis_path.name} \\
     --batch-dir {batch_dir.name} \\
     --skills-dir {skills_dir} \\
     --base {base_revision}

2. Read `{synthesis_path.name}` and **`PERF_KNOWLEDGE_BASE.md`** when it exists. Update pattern cards under `{knowledge_root.as_posix()}/references/patterns/` only.
3. Regenerate `{knowledge_root.as_posix()}/references/pattern_index.md`.
4. When `PERF_KNOWLEDGE_BASE.md` exists, run `{scripts_dir.as_posix()}/plan_workspaces_from_knowledge.py` and write `{batch_dir.as_posix()}/workspace-plan.json`. Scaffold **one workspace per plan entry**: directory and operator file named after `kernel_name`, one launch function per workspace (merge multiple kernels into one operator file when the same launch branches).
5. Plan and scaffold workspaces under `{batch_dir.as_posix()}` per `{workspace_scaffold_contract.as_posix()}` and the knowledge-base contract. Put copied helper `.py` modules under `deps/` — only one operator `.py` at each workspace root.
6. Run scaffold verification from repo root:

   triton-agent pattern-validation-verify -i {batch_dir.as_posix()}

   Fix every issue until it exits 0.
7. Record scaffold completion:

   python3 {scripts_dir.as_posix()}/record_iteration.py \\
     --state {state_path.as_posix()} --phase scaffold \\
     --note "workspaces ready for CLI optimize"

Rules:

- Do not run `triton-agent optimize-batch` (the CLI runs optimize after you finish).
- Do not edit staged backend install skills for knowledge updates.
- Do not copy entire multi-kernel `source_path` files when synthesis validates separate launch entrypoints.
"""


def build_analyze_prompt(
    *,
    repo_path: Path,
    batch_dir: Path,
    skills_workdir: Path,
    state_path: Path,
    audit_report_path: Path,
    iteration: int,
    max_iterations: int,
    skill_root: Path,
    knowledge_root: Path,
) -> str:
    iteration_contract = skill_root / "references" / "iteration-contract.md"
    scripts_dir = skill_root / "scripts"

    return f"""\
Analyze pattern-validation optimize evidence and decide whether the loop can complete.

Read:

  {iteration_contract.as_posix()}
  {skill_root.as_posix()}/SKILL.md
  {audit_report_path.as_posix()}

Repository root:

  {repo_path.as_posix()}

Batch root:

  {batch_dir.as_posix()}

Skills workdir:

  {skills_workdir.as_posix()}

Loop state:

  {state_path.as_posix()}

Current iteration: {iteration} / {max_iterations}

The evidence report aggregates `opt-round-*/attempts.md`, `summary.md`, and related artifacts.
`heuristic_suggested_pass` is a **hint only** (substring match on pattern IDs). You must judge whether optimize rounds actually applied synthesis-backed mechanisms.

Some workspaces may have failed optimize-batch runs; still review their partial `opt-round-*`
artifacts and `validation-meta.json` before deciding next steps.

Required steps:

1. Read `{audit_report_path.as_posix()}` and open round artifacts when excerpts are insufficient.
2. For each active workspace, decide pass/fail against synthesis and `validation-meta.json` `expected_patterns`.
3. If skills need fixes, edit `{knowledge_root.as_posix()}/references/patterns/`, regenerate `pattern_index.md`, and record a skill-update note.
4. When you are confident a workspace passed, archive it:

   python3 {scripts_dir.as_posix()}/audit_batch.py \\
     --batch-root {batch_dir.as_posix()} \\
     --archive-passed

   Use `--archive-passed` only for workspaces you reviewed; do not rely on heuristics alone.
5. If **all** validation targets are satisfied (`active_remaining` empty in a fresh evidence report), write `{batch_dir.as_posix()}/VALIDATION_SUMMARY.md` and:

   python3 {scripts_dir.as_posix()}/record_iteration.py \\
     --state {state_path.as_posix()} --phase complete \\
     --audit-report {audit_report_path.as_posix()} \\
     --note "pattern validation complete"

6. If more skill/optimize iterations are needed and iteration < {max_iterations}, record analysis without marking complete:

   python3 {scripts_dir.as_posix()}/record_iteration.py \\
     --state {state_path.as_posix()} --phase audit \\
     --audit-report {audit_report_path.as_posix()} \\
     --increment-iteration \\
     --note "needs another optimize iteration"

Rules:

- Do not run `triton-agent optimize-batch` (the CLI runs optimize between prepare/analyze phases).
- Do not hand-edit `pattern_index.md`.
- Do not delete `baseline/` when iterating skills.
"""
