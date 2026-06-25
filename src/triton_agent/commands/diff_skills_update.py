from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from triton_agent.diff_skills_update.models import DiffSkillsUpdateConfig, DiffSkillsUpdateSource
from triton_agent.diff_skills_update.workflow import run_diff_skills_update


def handle_diff_skills_update(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    try:
        config = _config_from_args(args)
        results = run_diff_skills_update(config, stream=sys.stderr)
    except ValueError as exc:
        parser.error(str(exc))
    failures = [result for result in results if not result.succeeded and result.status != "skipped"]
    for result in results:
        updated = ", ".join(result.updated_patterns) if result.updated_patterns else "none"
        print(
            f"{result.status}: {result.pair.operator_dir.name}/{result.pair.baseline_path.name} "
            f"updated_patterns=[{updated}] -> {result.report_path}",
            file=sys.stderr,
        )
    return 1 if failures else 0


def _config_from_args(args: argparse.Namespace) -> DiffSkillsUpdateConfig:
    input_root = Path(args.input).expanduser().resolve()
    skills_dir = (
        Path(args.skills_dir).expanduser().resolve()
        if getattr(args, "skills_dir", None)
        else input_root / "skills"
    )
    update_skills_dir = (
        Path(args.update_skills_dir).expanduser().resolve()
        if getattr(args, "update_skills_dir", None)
        else input_root / "update_skills"
    )
    concurrency = int(getattr(args, "concurrency", 1))
    if concurrency < 1:
        raise ValueError("--concurrency must be positive")
    max_iterations = int(getattr(args, "max_iterations", 3))
    if max_iterations < 1:
        raise ValueError("--max-iterations must be positive")
    source = cast(DiffSkillsUpdateSource, args.source)
    return DiffSkillsUpdateConfig(
        input_root=input_root,
        skills_dir=skills_dir,
        update_skills_dir=update_skills_dir,
        source=source,
        agent_name=str(getattr(args, "agent", "codex")),
        max_iterations=max_iterations,
        concurrency=concurrency,
        stream_output=bool(getattr(args, "stream_output", True)),
        verbose=bool(getattr(args, "verbose", False)),
        force=bool(getattr(args, "force", False)),
        skip_existing=bool(getattr(args, "skip_existing", False)),
        promote_converged_skills=bool(getattr(args, "promote_converged_skills", False)),
        base_revision=str(getattr(args, "git_base", None) or ""),
    )
