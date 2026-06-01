from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from triton_agent.backends.factory import create_runner
from triton_agent.pattern_validation_loop.git_worktree import resolve_git_worktree
from triton_agent.pattern_validation_loop.seed_skills import (
    DEFAULT_SKILLS_DIR_NAME,
    seed_pattern_validation_skills_dir,
)
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.prompts import append_additional_user_instructions
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager, staged_skill_dir
from triton_agent.skills_source_dir import OPTIMIZE_KNOWLEDGE_SKILL_NAME
from triton_agent.verbose import emit_verbose_lines


DEFAULT_SYNTHESIS_FILE = "PERF_PATTERN_SYNTHESIS.md"
DEFAULT_BATCH_DIR = "pattern-validation-batch"
DEFAULT_STATE_FILE = ".triton-agent/pattern-validation-loop-state.json"


def build_pattern_validation_loop_prompt(
    *,
    repo_path: Path,
    synthesis_path: Path,
    batch_dir: Path,
    skills_workdir: Path,
    skills_dir: str,
    state_path: Path,
    base_revision: str,
    min_rounds: int,
    max_iterations: int,
    agent_name: str,
    optimize_knowledge: str,
) -> str:
    backend_skills = staged_skill_dir(agent_name)
    skill_root = backend_skills / "triton-npu-pattern-validation-loop"
    knowledge_root = skills_workdir / OPTIMIZE_KNOWLEDGE_SKILL_NAME
    iteration_contract = skill_root / "references" / "iteration-contract.md"
    skill_update_contract = skill_root / "references" / "skill-update-contract.md"
    workspace_scaffold_contract = skill_root / "references" / "workspace-scaffold-contract.md"
    scripts_dir = skill_root / "scripts"

    return f"""\
Run the full pattern validation loop until all batch workspaces pass audit or `max_iterations` is reached.

Use the staged skill `triton-npu-pattern-validation-loop` as the **only** orchestration contract.
Read these references before acting:

  {iteration_contract.as_posix()}
  {skill_update_contract.as_posix()}
  {workspace_scaffold_contract.as_posix()}
  {skill_root.as_posix()}/SKILL.md

Repository root:

  {repo_path.as_posix()}

Synthesis report (input — read fully; do not assume fixed section IDs or tables):

  {synthesis_path.as_posix()}

Batch root (you create workspaces here):

  {batch_dir.as_posix()}

Persistent loop skills workdir (edit knowledge here; never delete this directory):

  {skills_workdir.as_posix()}

Loop state file:

  {state_path.as_posix()}

Git base revision for pre-optimization snapshots:

  {base_revision}

Optimize batch settings:

  min_rounds={min_rounds}
  optimize_knowledge={optimize_knowledge}

Loop limits:

  max_iterations={max_iterations}

Optimize knowledge to edit during the loop:

  {knowledge_root.as_posix()}

State and audit helpers (python3):

  {scripts_dir.as_posix()}/init_loop_state.py
  {scripts_dir.as_posix()}/record_iteration.py
  {scripts_dir.as_posix()}/reset_workspace_rounds.py
  {scripts_dir.as_posix()}/audit_batch.py
  {knowledge_root.as_posix()}/scripts/build_pattern_index.py

Optional scaffold helpers (only if you authored manifest JSON yourself):

  {scripts_dir.as_posix()}/scaffold_batch.py
  {scripts_dir.as_posix()}/generate_manifest.py

Required end-to-end behavior:

1. Initialize loop state if `{state_path.as_posix()}` does not exist (pass `--skills-dir {skills_dir}` to init_loop_state.py). The CLI already seeded `{skills_workdir.as_posix()}` from the install bundle when missing.
2. Read `{synthesis_path.as_posix()}` (and `PERF_KNOWLEDGE_BASE.md` if needed). **You** decide which lessons become pattern card edits under `{knowledge_root.as_posix()}/references/patterns/`.
3. Regenerate `{knowledge_root.as_posix()}/references/pattern_index.md` after each skill edit batch.
4. **You** plan validation workspaces: select operators, extract pre-opt snapshots with Git (`{base_revision}..HEAD`), find and copy tests/benches/dependencies, write each `validation-meta.json` under `{batch_dir.as_posix()}`. Follow `{workspace_scaffold_contract.as_posix()}`.
5. Run from `{repo_path.as_posix()}`:

   triton-agent optimize-batch -i {batch_dir.as_posix()} --resume fresh --reset-optimize --min-rounds {min_rounds} --max-concurrency 1 --show-output --optimize-knowledge {optimize_knowledge} --skills-source-dir {skills_dir} --agent {agent_name}

6. Audit with `{scripts_dir.as_posix()}/audit_batch.py --batch-root {batch_dir.as_posix()} --archive-passed --json` and save to `{batch_dir.as_posix()}/audit-report.json`. Passed workspaces move to `{batch_dir.as_posix()}/_completed/` and are skipped by later optimize-batch runs.
7. If any **active** workspace fails audit and iteration < {max_iterations}:
   - analyze missing patterns from attempts/summary
   - update pattern cards under `{knowledge_root.as_posix()}` + regenerate index
   - run `{scripts_dir.as_posix()}/reset_workspace_rounds.py --batch-root {batch_dir.as_posix()}`
   - re-run optimize-batch with `--resume continue --skills-source-dir {skills_dir}` per workspace or the whole batch as appropriate
   - re-audit
8. On full pass (`active_remaining` empty in audit report), write `{batch_dir.as_posix()}/VALIDATION_SUMMARY.md` and mark loop state complete.

Rules:

- Do not depend on `G1-I1` tables or blind `generate_manifest.py` output; interpret synthesis yourself.
- Do not promote repo-local-only or rejected lessons into skills or `expected_patterns`.
- Do not edit staged backend skills (for example `{backend_skills.as_posix()}/`) for knowledge updates; use `{skills_workdir.as_posix()}` only.
- Every optimize-batch run must pass `--skills-source-dir {skills_dir}`.
- Do not delete `baseline/` when iterating skills; use reset_workspace_rounds.py.
- Do not hand-edit `pattern_index.md`.
- Do not re-run optimize on workspaces already archived under `_completed/`.
- Treat optimize-batch success as necessary but not sufficient; audit must pass.
- Run real shell commands; do not fabricate audit JSON, workspaces, or round artifacts.
"""


def build_pattern_validation_loop_request(
    *,
    target_path: Path,
    synthesis_output: str = DEFAULT_SYNTHESIS_FILE,
    batch_dir: str = DEFAULT_BATCH_DIR,
    skills_dir: str = DEFAULT_SKILLS_DIR_NAME,
    base_revision: str = "origin/main",
    min_rounds: int = 10,
    max_iterations: int = 5,
    optimize_knowledge: str = "v1",
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    user_prompt: str | None = None,
) -> AgentRequest:
    repo_root = resolve_git_worktree(target_path)
    synthesis_path = _resolve_repo_path(repo_root, synthesis_output)
    if not synthesis_path.is_file():
        raise ValueError(
            f"Synthesis report not found: {synthesis_path}. "
            "Pass --synthesis to an existing synthesis report or create PERF_PATTERN_SYNTHESIS.md in the repo.",
        )
    batch_path = _resolve_repo_path(repo_root, batch_dir)
    state_path = repo_root / DEFAULT_STATE_FILE
    skills_workdir = seed_pattern_validation_skills_dir(
        repo_root,
        skills_dir,
        optimize_knowledge=optimize_knowledge,
    )
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.PATTERN_VALIDATION_LOOP,
        optimize_knowledge=optimize_knowledge,
    )
    prompt = append_additional_user_instructions(
        build_pattern_validation_loop_prompt(
            repo_path=repo_root,
            synthesis_path=synthesis_path,
            batch_dir=batch_path,
            skills_workdir=skills_workdir,
            skills_dir=skills_dir,
            state_path=state_path,
            base_revision=base_revision,
            min_rounds=min_rounds,
            max_iterations=max_iterations,
            agent_name=agent_name,
            optimize_knowledge=optimize_knowledge,
        ),
        user_prompt,
    )
    return AgentRequest(
        command_kind=CommandKind.PATTERN_VALIDATION_LOOP,
        input_path=repo_root,
        operator_path=None,
        output_path=batch_path,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=verbose,
        show_output=show_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="triton-npu-pattern-validation-loop",
        prompt=prompt,
        workdir=repo_root,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        min_rounds=min_rounds,
    )


def run_pattern_validation_loop(
    *,
    target_path: Path,
    synthesis_output: str = DEFAULT_SYNTHESIS_FILE,
    batch_dir: str = DEFAULT_BATCH_DIR,
    skills_dir: str = DEFAULT_SKILLS_DIR_NAME,
    base_revision: str = "origin/main",
    min_rounds: int = 10,
    max_iterations: int = 5,
    optimize_knowledge: Literal["v1", "v2", "v3"] = "v1",
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    user_prompt: str | None = None,
) -> int:
    try:
        request = build_pattern_validation_loop_request(
            target_path=target_path,
            synthesis_output=synthesis_output,
            batch_dir=batch_dir,
            skills_dir=skills_dir,
            base_revision=base_revision,
            min_rounds=min_rounds,
            max_iterations=max_iterations,
            optimize_knowledge=optimize_knowledge,
            agent_name=agent_name,
            verbose=verbose,
            show_output=show_output,
            user_prompt=user_prompt,
        )
    except ValueError as exc:
        print(f"[pattern-validation-loop] {exc}", file=sys.stderr, flush=True)
        return 2

    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        agent_name,
        request.workdir,
        skill_names=request.staged_skill_names,
        skill_sources=request.staged_skill_sources,
    )
    if verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))

    try:
        runner = create_runner(agent_name)
        result = runner.run(request)
    except FileNotFoundError as exc:
        print(
            f"[pattern-validation-loop] agent executable not found: {exc}. "
            f"Make sure the '{agent_name}' CLI is installed and available in PATH.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        cleanup_warnings = manager.cleanup(links)
        if cleanup_warnings:
            for warning in cleanup_warnings:
                print(f"[pattern-validation-loop] cleanup warning: {warning}", file=sys.stderr, flush=True)

    return result.return_code


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()
