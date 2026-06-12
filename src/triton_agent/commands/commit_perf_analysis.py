from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from triton_agent.commit_perf_analysis.launcher import run_commit_perf_analysis


def handle_analyze_commit_perf(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    target_path = Path(args.input).expanduser()
    target_chip = cast(Literal["A3", "A5"], getattr(args, "target_chip", "A5"))
    return run_commit_perf_analysis(
        target_path=target_path,
        output=getattr(args, "output", None),
        synthesis_output=getattr(args, "synthesis_output", None),
        base_revision=str(getattr(args, "base", "origin/main")),
        target_chip=target_chip,
        include_ir=bool(getattr(args, "include_ir", False)),
        force=bool(getattr(args, "force", False)),
        pull_requests=list(getattr(args, "pull_request", []) or []),
        agent_name=str(getattr(args, "agent", "codex")),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        user_prompt=getattr(args, "prompt", None),
    )


__all__ = ["handle_analyze_commit_perf"]
