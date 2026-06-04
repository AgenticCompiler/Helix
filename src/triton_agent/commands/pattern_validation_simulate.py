from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

from triton_agent.pattern_validation_loop.simulate_loop import run_pattern_validation_simulate_loop
from triton_agent.pattern_validation_loop.simulate_plan import build_simulate_plan_config


def handle_pattern_validation_simulate(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> int:
    optimize_knowledge = cast(Literal["v1", "v2", "v3"], getattr(args, "optimize_knowledge", "v1"))
    target_chip = cast(Literal["A3", "A5"] | None, getattr(args, "target_chip", None))
    test_mode = cast(Literal["standalone", "differential"] | None, getattr(args, "test_mode", None))
    bench_mode = cast(Literal["standalone", "msprof"] | None, getattr(args, "bench_mode", None))
    try:
        config = build_simulate_plan_config(
            target_path=Path(args.input).expanduser(),
            batch_dir=str(getattr(args, "batch_dir", "pattern-validation-batch")),
            skills_dir=str(getattr(args, "skills_dir", "pattern-validation-skills")),
            agent_name=str(getattr(args, "agent", "codex")),
            optimize_knowledge=optimize_knowledge,
            target_chip=target_chip or "A5",
            test_mode=test_mode,
            bench_mode=bench_mode,
            user_prompt=getattr(args, "prompt", None),
            verbose=bool(getattr(args, "verbose", False)),
            show_output=bool(getattr(args, "show_output", True)),
            skip_verify=bool(getattr(args, "skip_verify", False)),
            run_optimize_after=bool(getattr(args, "run_optimize", False)),
            max_iterations=int(getattr(args, "max_iterations", 5)),
            synthesis_output=str(getattr(args, "synthesis", "PERF_PATTERN_SYNTHESIS.md")),
            knowledge_base=str(getattr(args, "knowledge_base", "PERF_KNOWLEDGE_BASE.md")),
            base_revision=str(getattr(args, "base", "origin/main")),
            skip_launch_functions=list(getattr(args, "skip_launch", []) or []),
        )
    except ValueError as exc:
        print(f"[pattern-validation-simulate] {exc}", file=sys.stderr)
        return 2

    exit_code, report_path = run_pattern_validation_simulate_loop(config)
    if exit_code == 0:
        print(report_path.as_posix())
    return exit_code


__all__ = ["handle_pattern_validation_simulate"]
