from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

from triton_agent.report.workspace import generate_workspace_report


def handle_report_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
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
                generate_workspace_report,
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


__all__ = ["handle_report_batch"]
