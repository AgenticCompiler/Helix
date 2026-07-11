"""Per-workspace report generation via agent."""

from __future__ import annotations

import sys
from pathlib import Path

from helix.backends.factory import create_runner
from helix.models import AgentRequest, CommandKind
from helix.trace.core import new_trace_run_id
from helix.prompts import build_prompt
from helix.paths import skills_root
from helix.terminal.logs import show_output_log_path
from helix.skills.selection import resolve_staged_skills
from helix.skills.staging import SkillLinkManager


def build_hardware_info_text() -> str:
    from helix.report.hardware import capture_hardware_info

    hardware = capture_hardware_info()
    hw_info_lines: list[str] = []
    if hardware.get("chip_name"):
        hw_info_lines.append(f"- chip_name: {hardware['chip_name']}")
    if hardware.get("cann_version"):
        hw_info_lines.append(f"- cann_version: {hardware['cann_version']}")
    if hardware.get("driver_version"):
        hw_info_lines.append(f"- driver_version: {hardware['driver_version']}")
    return "\n".join(hw_info_lines)


def append_report_instructions(prompt: str, workspace: Path, hardware_info: str = "") -> str:
    hw_section = ""
    if hardware_info:
        hw_section = f"\nHardware environment information:\n{hardware_info}\n"
    return (
        f"{prompt}\n\n"
        f"{hw_section}"
        f"Your working directory is the operator workspace:\n\n"
        f"  {workspace.as_posix()}\n\n"
        f"Read the local skill `ascend-npu-report` from the workspace skills directory "
        f"as the primary workflow contract. Follow its steps to read the workspace "
        f"artifacts (opt-note.md, opt-round-*/summary.md, operator source, "
        f"round-state.json, etc.) and generate report.md in this directory.\n\n"
        f"The report must be in Chinese and follow the template in the skill's references/report-format.md."
    )


def build_report_request(
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
        stream_output=show_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="ascend-npu-report",
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
    hw_info = build_hardware_info_text()

    built_prompt = build_prompt(
        CommandKind.REPORT,
        workspace,
        None,
        None,
        None,
        None,
        False,
    )
    built_prompt = append_report_instructions(built_prompt, workspace, hw_info)

    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.REPORT,
    )

    request = build_report_request(
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
        if request.stream_output:
            detail = result.stderr.strip()
            if detail:
                return False, detail
            return False, f"agent execution failed; see show-output log: {show_output_log_path(request)}"
        detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
        return False, detail[:120]

    report_path = workspace / "report.md"
    if report_path.exists():
        return True, "report.md written"
    return False, "agent completed but report.md was not created"
