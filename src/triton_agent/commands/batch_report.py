from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

from triton_agent.backends.factory import create_runner
from triton_agent.batch_report.collector import write_batch_report_state
from triton_agent.batch_report.render import render_batch_report_file
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.output import render_result
from triton_agent.prompts import build_prompt
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager


def handle_batch_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")

    agent_name = getattr(args, "agent", "codex")
    show_output = bool(getattr(args, "show_output", False))
    report_workers = int(getattr(args, "report_workers", 4))

    print(
        f"[report-batch] start batch report: root={root.as_posix()}, agent={agent_name}",
        file=sys.stderr,
        flush=True,
    )

    # Phase 1: Existing batch-level report
    state_path = write_batch_report_state(root)
    print(f"Report-batch state written to: {state_path}", flush=True)

    report_path = render_batch_report_file(state_path)
    print(f"Report-batch written to: {report_path}", flush=True)

    # Phase 2: Per-workspace agent-driven report.md generation
    workspaces = _discover_workspaces(root)
    if not workspaces:
        print("[report-batch] no workspace directories found for per-workspace reports.", file=sys.stderr, flush=True)
        return 0

    print(
        f"\n[report-batch] generating per-workspace report.md for {len(workspaces)} workspace(s) "
        f"using agent={agent_name}, workers={report_workers}...",
        file=sys.stderr,
        flush=True,
    )

    ok_count = 0
    fail_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, report_workers)) as executor:
        futures = {
            executor.submit(
                _generate_workspace_report,
                ws_path,
                agent_name,
                show_output,
            ): ws_path
            for ws_path in workspaces
        }
        for future in concurrent.futures.as_completed(futures):
            ws_path = futures[future]
            try:
                ok, message = future.result()
            except Exception as exc:
                ok, message = False, str(exc)
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {ws_path.name}: {message}", flush=True)

    print(
        f"\n[report-batch] completed: {ok_count} succeeded, {fail_count} failed, {len(workspaces)} total",
        file=sys.stderr,
        flush=True,
    )

    return 0


def _discover_workspaces(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _generate_workspace_report(
    ws_path: Path,
    agent_name: str,
    show_output: bool,
) -> tuple[bool, str]:
    built_prompt = build_prompt(
        CommandKind.OPERATOR_REPORT,
        ws_path,
        None,
        None,
        None,
        None,
        False,
    )
    built_prompt = _append_report_instructions(built_prompt, ws_path)

    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.OPERATOR_REPORT,
    )

    request = AgentRequest(
        command_kind=CommandKind.OPERATOR_REPORT,
        input_path=ws_path,
        operator_path=None,
        output_path=None,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=False,
        show_output=show_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="triton-npu-report",
        prompt=built_prompt,
        workdir=ws_path,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
    )

    print(
        f"[report] start: {ws_path.name}",
        file=sys.stderr,
        flush=True,
    )

    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        agent_name,
        ws_path,
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

    report_path = ws_path / "report.md"
    if report_path.exists():
        return True, "report.md written"
    return False, "agent completed but report.md was not created"


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


__all__ = ["handle_batch_report"]
