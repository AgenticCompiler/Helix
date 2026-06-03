from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.status.core import (
    inspect_optimize_status_workspace,
    scan_optimize_status_workspaces,
    workspace_has_optimize_artifacts,
)
from triton_agent.status.render import render_optimize_status_results
from triton_agent.status.schema import warn_missing_sources, write_status_schema


def handle_status(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")

    want_schema = bool(getattr(args, "schema", False))
    is_single_workspace = workspace_has_optimize_artifacts(root)

    if is_single_workspace:
        results = [inspect_optimize_status_workspace(root, verbose=bool(getattr(args, "verbose", False)))]
        exit_code = render_optimize_status_results(
            results,
            output_format=str(getattr(args, "format", "text")),
        )
    else:
        workspace_candidates = sorted(path for path in root.iterdir() if path.is_dir())
        if not workspace_candidates:
            print(f"No operator workspaces found under {root}", file=sys.stderr)
            return 1
        results = scan_optimize_status_workspaces(root, verbose=bool(getattr(args, "verbose", False)))
        exit_code = render_optimize_status_results(results, output_format=str(getattr(args, "format", "text")))

    if want_schema and not is_single_workspace:
        schema_path = write_status_schema(root, results)
        print(f"[status-schema] written to: {schema_path}", file=sys.stderr, flush=True)
        warn_missing_sources(root)

    return exit_code


__all__ = ["handle_status"]
