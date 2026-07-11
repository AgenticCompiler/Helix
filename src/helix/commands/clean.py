from __future__ import annotations

import argparse
import sys
from pathlib import Path

from helix.clean.core import (
    clean_batch_root_artifacts,
    clean_workspace,
    discover_clean_workspaces,
    is_cleanable_workspace,
)


def handle_clean(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")

    workspaces = discover_clean_workspaces(root)
    if not workspaces:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    deep = bool(getattr(args, "deep", False))
    single_workspace = len(workspaces) == 1 and workspaces[0] == root and is_cleanable_workspace(root)
    for workspace in workspaces:
        clean_workspace(workspace, deep=deep)
    if not single_workspace:
        clean_batch_root_artifacts(root)
    return 0

