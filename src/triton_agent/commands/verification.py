from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

from triton_agent.verification.batch import run_verify_batch
from triton_agent.verification.core import VerifyOptions, prepare_verify_target, run_verify


def handle_verify(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    workspace = Path(args.input).expanduser().resolve()
    if not workspace.exists():
        parser.error(f"Input path does not exist: {workspace}")
    if not workspace.is_dir():
        parser.error(f"Input path is not a directory: {workspace}")

    options = VerifyOptions(
        phase=cast(Literal["all", "test", "bench"], str(getattr(args, "phase", "all"))),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        keep_remote_workdir=bool(getattr(args, "keep_remote_workdir", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )
    try:
        target = prepare_verify_target(workspace)
        result = run_verify(target, options)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Verification directory: {result.verify_dir}")
    print(f"State file: {result.state_path}")
    print(f"Return code: {result.return_code}")
    return result.return_code


def handle_verify_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    return run_verify_batch(
        root,
        force_verify=bool(getattr(args, "force_verify", False)),
        options=VerifyOptions(
            remote=getattr(args, "remote", None),
            remote_workdir=getattr(args, "remote_workdir", None),
            keep_remote_workdir=bool(getattr(args, "keep_remote_workdir", False)),
            verbose=bool(getattr(args, "verbose", False)),
        ),
    )
