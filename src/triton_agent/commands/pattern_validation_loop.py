from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from triton_agent.pattern_validation_loop.launcher import run_pattern_validation_loop


def handle_pattern_validation_loop(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    optimize_knowledge = cast(Literal["v1", "v2", "v3"], getattr(args, "optimize_knowledge", "v1"))
    if args.min_rounds < 1:
        parser.error("--min-rounds must be at least 1")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    return run_pattern_validation_loop(
        target_path=Path(args.input).expanduser(),
        synthesis_output=str(getattr(args, "synthesis", "PERF_PATTERN_SYNTHESIS.md")),
        batch_dir=str(getattr(args, "batch_dir", "pattern-validation-batch")),
        skills_dir=str(getattr(args, "skills_dir", "pattern-validation-skills")),
        base_revision=str(getattr(args, "base", "origin/main")),
        min_rounds=int(args.min_rounds),
        max_iterations=int(args.max_iterations),
        optimize_knowledge=optimize_knowledge,
        agent_name=str(getattr(args, "agent", "codex")),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        user_prompt=getattr(args, "prompt", None),
    )


__all__ = ["handle_pattern_validation_loop"]
