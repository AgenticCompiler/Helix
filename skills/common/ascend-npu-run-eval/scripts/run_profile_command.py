"""CLI orchestration for the profile-bench command."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from result_payload import ResultPayload


RunLocalProfile = Callable[..., tuple[ResultPayload, Optional[Path]]]
RunRemoteProfile = Callable[..., tuple[ResultPayload, Optional[Path], str]]
LoadProfileFunctions = Callable[[], tuple[RunLocalProfile, RunRemoteProfile]]
ResolvePath = Callable[[argparse.ArgumentParser, str, str], Path]
RenderResult = Callable[[ResultPayload, bool], None]
BuildProfileReport = Callable[[Path, Optional[str]], str]
ProfileHint = Callable[[Path], str]


@dataclass(frozen=True)
class RunProfileDependencies:
    load_profile_functions: LoadProfileFunctions
    resolve_existing_path: ResolvePath
    render_result: RenderResult
    build_profile_report: BuildProfileReport
    profile_hint: ProfileHint


def handle_run_profile_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    remote: str | None,
    remote_workdir: str | None,
    dependencies: RunProfileDependencies,
) -> int:
    run_local_profile, run_remote_profile = dependencies.load_profile_functions()
    bench_file = dependencies.resolve_existing_path(parser, args.bench_file, "Bench file")
    operator_file = dependencies.resolve_existing_path(parser, args.operator_file, "Operator file")
    workspace: str | None = None
    try:
        if remote is not None:
            result, profile_dir, workspace = run_remote_profile(
                bench_file,
                operator_file,
                remote,
                remote_workdir,
                case_id=args.case_id,
                kernel_name=args.kernel_name,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        else:
            result, profile_dir = run_local_profile(
                bench_file,
                operator_file,
                case_id=args.case_id,
                kernel_name=args.kernel_name,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    dependencies.render_result(result, remote is not None and args.verbose)
    print(f"Return code: {result['return_code']}")
    if profile_dir is not None:
        print(f"Profile directory: {profile_dir}")
        print(dependencies.build_profile_report(profile_dir, args.target_op))
        print(dependencies.profile_hint(profile_dir))
    if remote is not None and args.keep_remote_workdir:
        print(f"Remote workspace: {workspace}")
    return int(result["return_code"])
