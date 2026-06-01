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
OPTIMIZE_BATCH_ENV_PREFIX = "TRITON_AGENT_STALL_TIMEOUT_SECONDS=0 "


def build_optimize_batch_extra_flags(
    *,
    target_chip: str | None = None,
    test_mode: str | None = None,
    bench_mode: str | None = None,
) -> str:
    parts: list[str] = []
    if target_chip is not None:
        parts.append(f"--target-chip {target_chip}")
    if test_mode is not None:
        parts.append(f"--test-mode {test_mode}")
    if bench_mode is not None:
        parts.append(f"--bench-mode {bench_mode}")
    if not parts:
        return ""
    return " " + " ".join(parts)


def build_optimize_batch_shell_command(
    *,
    batch_dir: Path | str,
    skills_dir: str,
    min_rounds: int,
    optimize_knowledge: str,
    agent_name: str,
    extra_flags: str = "",
    resume: Literal["fresh", "continue"],
    reset_optimize: bool = False,
) -> str:
    reset_flag = " --reset-optimize" if reset_optimize else ""
    return (
        f"{OPTIMIZE_BATCH_ENV_PREFIX}triton-agent optimize-batch -i {Path(batch_dir).as_posix()}"
        f" --resume {resume}{reset_flag}"
        f" --min-rounds {min_rounds} --concurrency 1 --show-output"
        f" --optimize-knowledge {optimize_knowledge} --skills-source-dir {skills_dir}"
        f"{extra_flags} --agent {agent_name}"
    )


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
    target_chip: str | None = None,
    test_mode: str | None = None,
    bench_mode: str | None = None,
) -> str:
    optimize_extra_flags = build_optimize_batch_extra_flags(
        target_chip=target_chip,
        test_mode=test_mode,
        bench_mode=bench_mode,
    )
    optimize_batch_initial = build_optimize_batch_shell_command(
        batch_dir=batch_dir,
        skills_dir=skills_dir,
        min_rounds=min_rounds,
        optimize_knowledge=optimize_knowledge,
        agent_name=agent_name,
        extra_flags=optimize_extra_flags,
        resume="fresh",
        reset_optimize=True,
    )
    optimize_batch_continue = build_optimize_batch_shell_command(
        batch_dir=batch_dir,
        skills_dir=skills_dir,
        min_rounds=min_rounds,
        optimize_knowledge=optimize_knowledge,
        agent_name=agent_name,
        extra_flags=optimize_extra_flags,
        resume="continue",
    )
    optimize_setting_lines = [
        f"  min_rounds={min_rounds}",
        f"  optimize_knowledge={optimize_knowledge}",
    ]
    if target_chip is not None:
        optimize_setting_lines.append(f"  target_chip={target_chip}")
    if test_mode is not None:
        optimize_setting_lines.append(f"  test_mode={test_mode}")
    if bench_mode is not None:
        optimize_setting_lines.append(f"  bench_mode={bench_mode}")
    if target_chip is None and test_mode is None and bench_mode is None:
        optimize_setting_lines.append(
            "  target_chip/test_mode/bench_mode: omit from optimize-batch (use optimize defaults)"
        )
    optimize_settings = "\n".join(optimize_setting_lines)
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

{optimize_settings}

Loop limits:

  max_iterations={max_iterations}

Optimize knowledge to edit during the loop:

  {knowledge_root.as_posix()}

State and audit helpers (python3):

  {scripts_dir.as_posix()}/init_loop_state.py
  {scripts_dir.as_posix()}/record_iteration.py
  {scripts_dir.as_posix()}/reset_workspace_rounds.py
  {scripts_dir.as_posix()}/audit_batch.py
  {scripts_dir.as_posix()}/verify_batch_scaffold.py
  {knowledge_root.as_posix()}/scripts/build_pattern_index.py

Optional scaffold helpers (only if you authored manifest JSON yourself):

  {scripts_dir.as_posix()}/scaffold_batch.py
  {scripts_dir.as_posix()}/generate_manifest.py

Required end-to-end behavior:

1. Initialize loop state if `{state_path.as_posix()}` does not exist (pass `--skills-dir {skills_dir}` to init_loop_state.py). The CLI already seeded `{skills_workdir.as_posix()}` from the install bundle when missing.
2. Read `{synthesis_path.as_posix()}` (and `PERF_KNOWLEDGE_BASE.md` if needed). **You** decide which lessons become pattern card edits under `{knowledge_root.as_posix()}/references/patterns/`.
3. Regenerate `{knowledge_root.as_posix()}/references/pattern_index.md` after each skill edit batch.
4. **You** plan validation workspaces: select operators, extract pre-opt snapshots with Git (`{base_revision}..HEAD`), find and copy tests/benches/dependencies, write each `validation-meta.json` under `{batch_dir.as_posix()}`. Follow `{workspace_scaffold_contract.as_posix()}`.
   - When synthesis validates multiple independent targets from the same repo `source_path`, you **must** follow **Step 2b (manual split)**: one workspace per launch entrypoint, operator file is a **minimal extract** — never `git show` the whole multi-kernel file into one workspace.
   - For each split workspace set `validation_target`, `split_from`, `included_symbols`, and `excluded_targets` in `validation-meta.json`.
   - Before optimize-batch, run `{scripts_dir.as_posix()}/verify_batch_scaffold.py --batch-root {batch_dir.as_posix()}` and fix every reported issue.
5. Run from `{repo_path.as_posix()}`:

   {optimize_batch_initial}

6. Audit with `{scripts_dir.as_posix()}/audit_batch.py --batch-root {batch_dir.as_posix()} --archive-passed --json` and save to `{batch_dir.as_posix()}/audit-report.json`. Passed workspaces move to `{batch_dir.as_posix()}/_completed/` and are skipped by later optimize-batch runs.
7. If any **active** workspace fails audit and iteration < {max_iterations}:
   - analyze missing patterns from attempts/summary
   - update pattern cards under `{knowledge_root.as_posix()}` + regenerate index
   - run `{scripts_dir.as_posix()}/reset_workspace_rounds.py --batch-root {batch_dir.as_posix()}`
   - re-run optimize-batch:

   {optimize_batch_continue}

   - re-audit
8. On full pass (`active_remaining` empty in audit report), write `{batch_dir.as_posix()}/VALIDATION_SUMMARY.md` and mark loop state complete.

Rules:

- Do not depend on `G1-I1` tables or blind `generate_manifest.py` output; interpret synthesis yourself.
- Do not promote repo-local-only or rejected lessons into skills or `expected_patterns`.
- Do not edit staged backend skills (for example `{backend_skills.as_posix()}/`) for knowledge updates; use `{skills_workdir.as_posix()}` only.
- Do not copy an entire multi-kernel repo `source_path` into a workspace when synthesis validates only one launch entrypoint inside it; use Step 2b manual extract.
- Do not run optimize-batch until `verify_batch_scaffold.py --batch-root {batch_dir.as_posix()}` passes for all active workspaces.
- Every optimize-batch run must pass `--skills-source-dir {skills_dir}`.
- Every optimize-batch shell command must prefix `TRITON_AGENT_STALL_TIMEOUT_SECONDS=0` so nested optimize agents do not stall-kill on long silent runs.
- Every optimize-batch run must pass `--show-output` so nested optimize agent logs stream to the terminal; long silent runs may be killed by job timeouts or watchdogs.
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
    target_chip: str | None = None,
    test_mode: str | None = None,
    bench_mode: str | None = None,
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
            target_chip=target_chip,
            test_mode=test_mode,
            bench_mode=bench_mode,
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
    target_chip: Literal["A3", "A5"] | None = None,
    test_mode: Literal["standalone", "differential"] | None = None,
    bench_mode: Literal["standalone", "msprof"] | None = None,
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
            target_chip=target_chip,
            test_mode=test_mode,
            bench_mode=bench_mode,
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
