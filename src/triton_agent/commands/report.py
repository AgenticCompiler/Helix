from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.backends.factory import create_runner
from triton_agent.models import CommandKind
from triton_agent.terminal.render import render_result
from triton_agent.prompts import append_additional_user_instructions, build_prompt
from triton_agent.paths import skills_root
from triton_agent.report.workspace import (
    append_report_instructions,
    build_hardware_info_text,
    build_report_request,
)
from triton_agent.terminal.logs import show_output_log_path
from triton_agent.skills.selection import resolve_staged_skills
from triton_agent.skills.staging import SkillLinkManager
from triton_agent.terminal.verbose import emit_verbose, emit_verbose_lines


def handle_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    workspace = Path(args.input).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"Not a directory: {workspace}")

    agent_name = getattr(args, "agent", "codex")
    interact = bool(getattr(args, "interact", False))
    stream_output = bool(getattr(args, "stream_output", True))
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
    hardware_info = build_hardware_info_text()
    built_prompt = append_report_instructions(built_prompt, workspace, hardware_info)

    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.REPORT,
    )

    request = build_report_request(
        workspace=workspace,
        agent_name=agent_name,
        prompt=built_prompt,
        interact=interact,
        verbose=verbose,
        show_output=stream_output,
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

    render_result(result, skip_stdout=stream_output)
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
        if stream_output:
            detail = result.stderr.strip() or f"agent execution failed; see show-output log: {show_output_log_path(request)}"
        else:
            detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
        print(f"[report] failed: {detail}", file=sys.stderr, flush=True)
    return result.return_code
