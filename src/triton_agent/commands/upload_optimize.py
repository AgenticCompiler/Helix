from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.optimize_upload.workflow import upload_optimize_workspace


def handle_upload_optimize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        parser.error(f"Input path must be a directory: {input_path}")

    try:
        response = upload_optimize_workspace(input_path, url=args.url, verbose=args.verbose)
    except ValueError as exc:
        print(f"Upload error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        return 1

    print(f"Uploaded {response.workspace_name} -> {Path(response.stored_path).name}")
    if args.verbose:
        print(f"  UID: {response.upload_uid}")
        print(f"  Timestamp: {response.upload_timestamp}")

    return 0
