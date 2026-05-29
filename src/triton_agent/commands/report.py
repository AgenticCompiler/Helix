from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.output import render_result
from triton_agent.prompts import append_additional_user_instructions, build_prompt
from triton_agent.otel_trace import new_trace_run_id
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager
from triton_agent.verbose import emit_verbose, emit_verbose_lines


def handle_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    workspace = Path(args.input).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"Not a directory: {workspace}")

    agent_name = getattr(args, "agent", "codex")
    interact = bool(getattr(args, "interact", False))
    show_output = bool(getattr(args, "show_output", False))
    verbose = bool(getattr(args, "verbose", False))
    user_prompt = getattr(args, "prompt", None)

    built_prompt = build_prompt(
        CommandKind.REPORT,
        workspace,
        None,
        None,
        None,
        None,
        False,
    )
    built_prompt = append_additional_user_instructions(built_prompt, user_prompt)
    # Explicitly instruct the agent to write report.md in the workspace.
    built_prompt = _append_report_instructions(built_prompt, workspace)

    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.REPORT,
    )

    request = _build_report_request(
        workspace=workspace,
        agent_name=agent_name,
        prompt=built_prompt,
        interact=interact,
        verbose=verbose,
        show_output=show_output,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
    )

    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        agent_name,
        workspace,
        skill_names=staged_skill_names,
        skill_sources=staged_skill_sources,
    )
    verbose_stream = sys.stderr
    if verbose:
        emit_verbose_lines(verbose_stream, "skills", manager.describe_prepare(links))
    print(
        f"[report] start report generation: workspace={workspace.as_posix()}, agent={agent_name}",
        file=sys.stderr,
        flush=True,
    )
    try:
        runner = create_runner(agent_name)
    except ValueError as exc:
        print(f"[report] invalid agent: {exc}", file=sys.stderr, flush=True)
        manager.cleanup(links)
        return 2
    try:
        result = runner.run(request)
    except FileNotFoundError as exc:
        print(
            f"[report] agent executable not found: {exc}. "
            f"Make sure the '{agent_name}' CLI is installed and available in PATH.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if verbose:
            emit_verbose_lines(verbose_stream, "skills", manager.describe_cleanup(links))
        cleanup_warnings = manager.cleanup(links)
        for warning in cleanup_warnings:
            emit_verbose(verbose_stream, "skills", warning)

    render_result(result, show_output=show_output)
    if result.succeeded:
        report_path = workspace / "report.md"
        if report_path.exists():
            print(
                f"[report] completed: report.md generated at {report_path.as_posix()}",
                file=sys.stderr,
                flush=True,
            )
            print(f"Report written to: {report_path}", flush=True)
        else:
            print("[report] warning: agent completed but report.md was not created", file=sys.stderr, flush=True)
    else:
        detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
        print(f"[report] failed: {detail}", file=sys.stderr, flush=True)
    return result.return_code


def _append_report_instructions(prompt: str, workspace: Path) -> str:
    return (
        f"{prompt}\n\n"
        f"Your working directory is the operator workspace:\n\n"
        f"  {workspace.as_posix()}\n\n"
        f"Read the local skill `triton-npu-report` from the workspace skills directory "
        f"as the primary workflow contract. Follow its steps to read the workspace "
        f"artifacts (env-info.json, opt-note.md, opt-round-*/summary.md, operator source, "
        f"round-state.json, etc.) and generate report.md in this directory.\n\n"
        f"The report must be in Chinese and follow the template in the skill's references/report-format.md."
    )


def _build_report_request(
    *,
    workspace: Path,
    agent_name: str,
    prompt: str,
    interact: bool,
    verbose: bool,
    show_output: bool,
    staged_skill_names: tuple[str, ...] | None,
    staged_skill_sources: dict[str, str] | None,
) -> AgentRequest:
    return AgentRequest(
        command_kind=CommandKind.REPORT,
        input_path=workspace,
        operator_path=None,
        output_path=None,
        test_mode=None,
        bench_mode=None,
        interact=interact,
        verbose=verbose,
        show_output=show_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="triton-npu-report",
        prompt=prompt,
        workdir=workspace,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        run_id=new_trace_run_id(prefix="report"),
    )


def generate_workspace_report(
    workspace: Path,
    agent_name: str,
    show_output: bool = False,
) -> tuple[bool, str]:
    built_prompt = build_prompt(
        CommandKind.REPORT,
        workspace,
        None,
        None,
        None,
        None,
        False,
    )
    built_prompt = _append_report_instructions(built_prompt, workspace)

    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.REPORT,
    )

    request = _build_report_request(
        workspace=workspace,
        agent_name=agent_name,
        prompt=built_prompt,
        interact=False,
        verbose=False,
        show_output=show_output,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
    )

    print(
        f"[report] start: {workspace.name}",
        file=sys.stderr,
        flush=True,
    )

    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        agent_name,
        workspace,
        skill_names=staged_skill_names,
        skill_sources=staged_skill_sources,
    )
    try:
        runner = create_runner(agent_name)
    except ValueError as exc:
        manager.cleanup(links)
        return False, f"invalid agent: {exc}"

    try:
        result = runner.run(request)
    except FileNotFoundError as exc:
        manager.cleanup(links)
        return False, f"agent executable not found: {exc}"
    except Exception as exc:
        manager.cleanup(links)
        return False, str(exc)
    finally:
        manager.cleanup(links)

    if not result.succeeded:
        detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
        return False, detail[:120]

    report_path = workspace / "report.md"
    if report_path.exists():
        return True, "report.md written"
    return False, "agent completed but report.md was not created"


__all__ = ["generate_workspace_report", "handle_report"]
