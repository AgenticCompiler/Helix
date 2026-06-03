from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.pattern_validation_loop.paths import resolve_repo_path
from triton_agent.pattern_validation_loop.workspace_plan import (
    DEFAULT_KNOWLEDGE_FILE,
    DEFAULT_WORKSPACE_PLAN_NAME,
    generate_workspace_plan,
    resolve_knowledge_base_path,
)


def handle_pattern_validation_plan(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> int:
    repo_root = Path(args.input).expanduser().resolve()
    if not repo_root.is_dir():
        parser.error(f"Input path is not a directory: {repo_root}")

    knowledge_path = resolve_knowledge_base_path(
        repo_root,
        str(getattr(args, "knowledge", DEFAULT_KNOWLEDGE_FILE)),
    )
    if not knowledge_path.is_file():
        parser.error(f"Knowledge base not found: {knowledge_path}")

    batch_dir = str(getattr(args, "batch_dir", "pattern-validation-batch"))
    output_value = str(getattr(args, "output", "")).strip()
    if output_value:
        output_path = Path(output_value).expanduser()
        if not output_path.is_absolute():
            output_path = repo_root / output_path
    else:
        output_path = resolve_repo_path(repo_root, batch_dir) / DEFAULT_WORKSPACE_PLAN_NAME

    try:
        payload, warnings = generate_workspace_plan(
            repo_root=repo_root,
            knowledge_path=knowledge_path,
            output_path=output_path,
            base_revision=str(getattr(args, "base", "")),
        )
    except RuntimeError as exc:
        print(f"[pattern-validation-plan] {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if payload is None:
        return 1

    print(output_path.as_posix())
    return 0


__all__ = ["handle_pattern_validation_plan"]
