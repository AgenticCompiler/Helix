from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.prompts import append_additional_user_instructions
from triton_agent.resources import skills_root
from triton_agent.skill_loader import load_skill_script_module
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager, staged_skill_dir
from triton_agent.verbose import emit_verbose_lines


DEFAULT_BASE_REVISION = "origin/main"
DEFAULT_OUTPUT_FILE = "PERF_KNOWLEDGE_BASE.md"
DEFAULT_SYNTHESIS_OUTPUT_FILE = "PERF_PATTERN_SYNTHESIS.md"


def _pull_request_helpers():
    return load_skill_script_module("triton-npu-analyze-commit-perf", "knowledge_pull_requests")


def resolve_pull_request_filter(pull_requests: list[str] | None) -> set[int] | None:
    if not pull_requests:
        return None
    return _pull_request_helpers().parse_pull_request_ids(pull_requests)


def format_pull_request_summary(pull_request_filter: set[int] | None) -> str:
    if not pull_request_filter:
        return "all commits in range"
    ordered = ", ".join(f"!{item}" for item in sorted(pull_request_filter))
    return f"merge requests {ordered}"


def build_collect_context_args(
    *,
    base_revision: str,
    pull_request_filter: set[int] | None,
) -> str:
    args = f"--base {base_revision} --output .triton-agent/commit-perf-context.json"
    if pull_request_filter is None:
        return args
    pr_args = " ".join(f"--pull-request {item}" for item in sorted(pull_request_filter))
    return f"{args} {pr_args}"


def count_filtered_commits_in_range(
    repo_root: Path,
    base_revision: str,
    pull_request_filter: set[int],
) -> int:
    pr_mod = _pull_request_helpers()
    result = _run_git(["rev-list", f"{base_revision}..HEAD"], cwd=repo_root)
    if result.returncode != 0:
        detail = result.stderr.strip() or "git rev-list failed"
        raise ValueError(f"Failed to list commits in {base_revision}..HEAD: {detail}")
    shas = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    commit_to_pr = pr_mod.build_commit_pull_request_map(repo_root, base_revision)
    filtered = pr_mod.filter_commit_shas_by_pull_requests(
        shas,
        pull_request_filter=pull_request_filter,
        commit_to_pr=commit_to_pr,
    )
    return len(filtered)


def build_commit_perf_analysis_prompt(
    *,
    repo_path: Path,
    output_path: Path,
    synthesis_output_path: Path,
    base_revision: str = DEFAULT_BASE_REVISION,
    target_chip: Literal["A3", "A5"] = "A5",
    include_ir: bool = False,
    force: bool = False,
    pull_request_filter: set[int] | None = None,
    agent_name: str = "codex",
) -> str:
    normalized_repo = repo_path.resolve()
    normalized_output = output_path.resolve()
    normalized_synthesis_output = synthesis_output_path.resolve()
    backend_skills = staged_skill_dir(agent_name)
    skill_root = backend_skills / "triton-npu-analyze-commit-perf"
    helper_path = skill_root / "scripts" / "collect_commit_context.py"
    group_path = skill_root / "scripts" / "group_commit_context_by_file.py"
    output_contract = skill_root / "references" / "output-contract.md"
    incremental_contract = skill_root / "references" / "incremental-file-analysis.md"
    synthesis_contract = skill_root / "references" / "pattern-synthesis-contract.md"
    pattern_index_path = backend_skills / "triton-npu-optimize-knowledge" / "references" / "pattern_index.md"
    knowledge_root = backend_skills / "triton-npu-optimize-knowledge"
    ir_instruction = (
        "IR support is enabled. Use the staged `triton-npu-analyze-ir` skill only when "
        "local IR artifacts are already available or can be collected safely without "
        "changing repository state."
        if include_ir
        else "IR support is disabled for this run. Leave IR fields empty when no IR evidence is available."
    )
    overwrite_instruction = (
        "The user allowed overwriting the report if it already exists."
        if force
        else "Do not overwrite an existing report; the CLI already checked that the requested output is free."
    )
    pull_request_instruction = (
        "Analyze only commits mapped to the selected merge requests. "
        f"Record them in `## Run Summary` as `| Analyzed pull requests | {', '.join(str(item) for item in sorted(pull_request_filter))} |`."
        if pull_request_filter
        else "Analyze all commits in the base..HEAD range unless hard-skipped by the helper."
    )
    collect_args = build_collect_context_args(
        base_revision=base_revision,
        pull_request_filter=pull_request_filter,
    )
    return f"""\
Analyze Git commits in the current repository for Triton Ascend NPU performance knowledge.

Use the local skill `triton-npu-analyze-commit-perf` from the workspace skills directory as the primary workflow contract.
Read the output contract before writing the report:

  {output_contract.as_posix()}

Repository root:

  {normalized_repo.as_posix()}

Base revision:

  {base_revision}

Target chip:

  {target_chip}

Commit scope:

  {format_pull_request_summary(pull_request_filter)}

Incremental report (file-by-file working document):

  {normalized_output.as_posix()}

Final synthesis report (pattern clustering + pattern-index alignment):

  {normalized_synthesis_output.as_posix()}

Pattern synthesis contract:

  {synthesis_contract.as_posix()}

Staged pattern index for comparison:

  {pattern_index_path.as_posix()}

Git context helper:

  {helper_path.as_posix()}

File grouping helper:

  {group_path.as_posix()}

Incremental workflow contract:

  {incremental_contract.as_posix()}

Generic optimization knowledge:

  {knowledge_root.as_posix()}

Run the workflow incrementally by file, not as one giant branch-wide analysis.

Steps:
1. Run `collect_commit_context.py` with `{collect_args}`.
2. If `commit_count` is zero, stop with an actionable error. Do not write a placeholder report.
3. Run `group_commit_context_by_file.py` on the context JSON.
4. Initialize `{normalized_output.as_posix()}` with the report skeleton and write all hard-skipped commits.
5. Create `.triton-agent/commit-perf-analysis-state.json` listing pending file groups.
6. Analyze exactly one file group per round. For each file, analyze all commits that touched it, in chronological order, then append that file section to `## File Analyses` and update the state file before moving on.
7. After every file group is done, update the incremental report sections: Reusable Rules, Pattern Promotion Candidates, and Limitations.
8. Run the mandatory final pattern synthesis round from `{synthesis_contract.as_posix()}`:
   - Read the completed incremental report at `{normalized_output.as_posix()}`
   - Cluster similar lessons into consolidated pattern groups with per-item detail
   - Compare against `{pattern_index_path.as_posix()}`
   - For each item, recommend whether skills should be updated (`no-change`, `extend-existing-card`, `promote-new-pattern-card`, `local-only`, `reject`)
   - Write the consolidated report to `{normalized_synthesis_output.as_posix()}`
   - Do not edit pattern cards or `pattern_index.md` in this workflow

Keep helper hard-skipped commits visible only in `## Skipped Commits` (one line each).
Do not write performance-unrelated commits to the report: omit `correctness-related`, `noise`, formatting-only, docs/test/ci-only, and similar commits from `## File Analyses` and from synthesis sections.
Write timeline entries only for `performance-related` and `rollback-or-negative` commits.
Treat rollback or negative-result commits as failed optimization lessons, not as noise.
Omit entire file sections when a file has no performance-relevant commits after soft classification.
{pull_request_instruction}
{ir_instruction}
{overwrite_instruction}

Incremental report path:

  {normalized_output.as_posix()}

Final synthesis report path:

  {normalized_synthesis_output.as_posix()}"""


def build_commit_perf_analysis_request(
    *,
    target_path: Path,
    output: str | None = None,
    synthesis_output: str | None = None,
    base_revision: str = DEFAULT_BASE_REVISION,
    target_chip: Literal["A3", "A5"] = "A5",
    include_ir: bool = False,
    force: bool = False,
    pull_requests: list[str] | None = None,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    user_prompt: str | None = None,
) -> AgentRequest:
    repo_root = resolve_git_worktree(target_path)
    validate_base_revision(repo_root, base_revision)
    pull_request_filter = resolve_pull_request_filter(pull_requests)
    commit_count = count_commits_in_range(repo_root, base_revision)
    if commit_count == 0:
        raise ValueError(
            f"No commits found in range {base_revision}..HEAD. "
            "Check out the branch you want to analyze or choose an earlier --base revision."
        )
    if pull_request_filter is not None:
        filtered_count = count_filtered_commits_in_range(
            repo_root,
            base_revision,
            pull_request_filter,
        )
        if filtered_count == 0:
            ordered = ", ".join(str(item) for item in sorted(pull_request_filter))
            raise ValueError(
                f"No commits in {base_revision}..HEAD matched pull request filter: {ordered}. "
                "Check --base and --pull-request values."
            )
    output_path = resolve_report_path(repo_root, output)
    synthesis_output_path = resolve_report_path(repo_root, synthesis_output or DEFAULT_SYNTHESIS_OUTPUT_FILE)
    if output_path.exists() and not force:
        raise ValueError(
            f"Output report already exists: {output_path}. "
            "Pass --force to overwrite it or choose a different --output path."
        )
    if synthesis_output_path.exists() and not force:
        raise ValueError(
            f"Synthesis report already exists: {synthesis_output_path}. "
            "Pass --force to overwrite it or choose a different --synthesis-output path."
        )
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.ANALYZE_COMMIT_PERF,
        include_ir=include_ir,
    )
    prompt = append_additional_user_instructions(
        build_commit_perf_analysis_prompt(
            repo_path=repo_root,
            output_path=output_path,
            synthesis_output_path=synthesis_output_path,
            base_revision=base_revision,
            target_chip=target_chip,
            include_ir=include_ir,
            force=force,
            pull_request_filter=pull_request_filter,
            agent_name=agent_name,
        ),
        user_prompt,
    )
    return AgentRequest(
        command_kind=CommandKind.ANALYZE_COMMIT_PERF,
        input_path=repo_root,
        operator_path=None,
        output_path=synthesis_output_path,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=verbose,
        show_output=show_output,
        force_overwrite=force,
        agent_name=agent_name,
        skill_name="triton-npu-analyze-commit-perf",
        prompt=prompt,
        workdir=repo_root,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        target_chip=target_chip,
    )


def run_commit_perf_analysis(
    *,
    target_path: Path,
    output: str | None = None,
    synthesis_output: str | None = None,
    base_revision: str = DEFAULT_BASE_REVISION,
    target_chip: Literal["A3", "A5"] = "A5",
    include_ir: bool = False,
    force: bool = False,
    pull_requests: list[str] | None = None,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    user_prompt: str | None = None,
) -> int:
    try:
        request = build_commit_perf_analysis_request(
            target_path=target_path,
            output=output,
            synthesis_output=synthesis_output,
            base_revision=base_revision,
            target_chip=target_chip,
            include_ir=include_ir,
            force=force,
            pull_requests=pull_requests,
            agent_name=agent_name,
            verbose=verbose,
            show_output=show_output,
            user_prompt=user_prompt,
        )
    except ValueError as exc:
        print(f"[commit-perf-analysis] {exc}", file=sys.stderr, flush=True)
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
            f"[commit-perf-analysis] agent executable not found: {exc}. "
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
            emit_verbose_lines(sys.stderr, "skills", cleanup_warnings)

    if not result.succeeded:
        detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
        print(f"[commit-perf-analysis] analysis failed: {detail}", file=sys.stderr, flush=True)
        return result.return_code if result.return_code != 0 else 1

    incremental_path = resolve_report_path(request.workdir, output)
    synthesis_path = resolve_report_path(
        request.workdir,
        synthesis_output or DEFAULT_SYNTHESIS_OUTPUT_FILE,
    )
    if not incremental_path.is_file():
        print(
            "[commit-perf-analysis] analysis completed but incremental report was not created: "
            + incremental_path.as_posix(),
            file=sys.stderr,
            flush=True,
        )
        return 1
    if synthesis_path is None or not synthesis_path.is_file():
        missing = synthesis_path.as_posix() if synthesis_path is not None else DEFAULT_SYNTHESIS_OUTPUT_FILE
        print(
            "[commit-perf-analysis] analysis completed but synthesis report was not created: "
            + missing,
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        "[commit-perf-analysis] analysis completed:\n"
        f"  incremental: {incremental_path.as_posix()}\n"
        f"  synthesis: {synthesis_path.as_posix()}",
        file=sys.stderr,
        flush=True,
    )
    return 0


def resolve_git_worktree(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    cwd = candidate if candidate.is_dir() else candidate.parent
    if not cwd.exists():
        raise ValueError(f"Input path does not exist: {candidate}")
    result = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0:
        detail = result.stderr.strip() or "not a Git work tree"
        raise ValueError(f"Input path is not inside a Git work tree: {candidate} ({detail})")
    return Path(result.stdout.strip()).resolve()


def validate_base_revision(repo_root: Path, base_revision: str) -> None:
    if not base_revision.strip():
        raise ValueError("--base must not be empty")
    result = _run_git(["rev-parse", "--verify", f"{base_revision}^{{commit}}"], cwd=repo_root)
    if result.returncode != 0:
        detail = result.stderr.strip() or "revision not found"
        raise ValueError(f"Base revision is not a valid commit: {base_revision} ({detail})")


def count_commits_in_range(repo_root: Path, base_revision: str) -> int:
    result = _run_git(["rev-list", "--count", f"{base_revision}..HEAD"], cwd=repo_root)
    if result.returncode != 0:
        detail = result.stderr.strip() or "git rev-list failed"
        raise ValueError(f"Failed to count commits in {base_revision}..HEAD: {detail}")
    try:
        return int(result.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"Unexpected rev-list output: {result.stdout!r}") from exc


def resolve_report_path(repo_root: Path, output: str | None) -> Path:
    if output is None or not output.strip():
        return repo_root / DEFAULT_OUTPUT_FILE
    output_path = Path(output).expanduser()
    if output_path.is_absolute():
        return output_path.resolve()
    return (repo_root / output_path).resolve()


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
