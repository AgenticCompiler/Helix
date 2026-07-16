"""Command flow for the three public run-test subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Protocol, TextIO, cast

from compare_result import (
    compare_remote_result_files,
    compare_result_files,
    compare_result_payload_objects,
    find_case_result_payload,
    load_case_result_payload,
)
from execution_lifecycle import (
    active_optimize_round_context,
    append_optimize_timing_event,
    cleanup_run_test_pt_files,
)
from result_payload import ResultPayload
from run_test_api import (
    parse_test_metadata,
    run_local_test,
    run_local_test_case_payload,
    run_remote_differential_comparison,
    run_remote_test,
    run_remote_test_case_payload,
)


class RunTestDependencies:
    def __init__(
        self,
        *,
        parse_test_metadata: Any,
        run_local_test: Any,
        run_remote_test: Any,
        run_remote_differential_comparison: Any,
        run_local_test_case_payload: Any,
        run_remote_test_case_payload: Any,
        load_case_result_payload: Any,
        find_case_result_payload: Any,
        compare_result_payload_objects: Any,
        compare_result_files: Any,
        compare_remote_result_files: Any,
    ) -> None:
        self.parse_test_metadata = parse_test_metadata
        self.run_local_test = run_local_test
        self.run_remote_test = run_remote_test
        self.run_remote_differential_comparison = run_remote_differential_comparison
        self.run_local_test_case_payload = run_local_test_case_payload
        self.run_remote_test_case_payload = run_remote_test_case_payload
        self.load_case_result_payload = load_case_result_payload
        self.find_case_result_payload = find_case_result_payload
        self.compare_result_payload_objects = compare_result_payload_objects
        self.compare_result_files = compare_result_files
        self.compare_remote_result_files = compare_remote_result_files


class CompareRemoteResultFn(Protocol):
    def __call__(
        self,
        ref_result: Path,
        new_result: Path,
        remote: str,
        remote_workdir: str | None,
        *,
        accuracy_mode: str | None = None,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> int: ...


def handle_run_test_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    remote: str | None,
    remote_workdir: str | None,
    dependencies: RunTestDependencies | None = None,
) -> int:
    dependencies = dependencies or _default_dependencies()
    test_file = _resolve_existing_path(parser, args.test_file, "Test file")
    operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
    timing_context = (
        active_optimize_round_context(test_file, operator_file)
        if args.command == "run-test-optimize"
        else None
    )
    ref_result = _resolve_optional_existing_path(
        parser, getattr(args, "ref_result", None), "Reference result"
    )
    ref_operator_file = _resolve_optional_existing_path(
        parser, getattr(args, "ref_operator_file", None), "Reference operator file"
    )
    parse_test_metadata_fn = dependencies.parse_test_metadata
    run_local_test_fn = dependencies.run_local_test
    run_remote_test_fn = dependencies.run_remote_test
    resolved_test_mode = args.test_mode or _resolve_test_mode_from_metadata(
        test_file, parse_test_metadata_fn
    )
    case_id = cast(Optional[str], getattr(args, "case_id", None))
    accuracy_mode = cast(Optional[str], getattr(args, "accuracy_mode", None))
    require_reference_input = args.command in {"run-test-convert", "run-test-optimize"}

    if remote is not None and resolved_test_mode == "differential":
        _validate_remote_differential_inputs(parser, ref_result, ref_operator_file)
        assert ref_operator_file is not None
        try:
            result, remote_workspace = dependencies.run_remote_differential_comparison(
                test_file,
                ref_operator_file,
                operator_file,
                remote,
                remote_workdir,
                case_id=case_id,
                accuracy_mode=accuracy_mode,
                keep_remote_workdir=bool(args.keep_remote_workdir),
                verbose=bool(args.verbose),
                stderr=sys.stderr,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _render_result(result, skip_stdout=True)
        print(f"Return code: {result['return_code']}")
        if args.keep_remote_workdir:
            print(f"Remote workspace: {remote_workspace}")
        return int(result["return_code"])

    if case_id is not None:
        _validate_run_test_comparison_inputs(
            parser,
            args.command,
            resolved_test_mode,
            ref_result,
            ref_operator_file,
            case_id=case_id,
            require_reference_input=require_reference_input,
        )
        run_local_payload_fn = dependencies.run_local_test_case_payload
        run_remote_payload_fn = dependencies.run_remote_test_case_payload
        load_case_payload_fn = dependencies.load_case_result_payload
        find_case_payload_fn = dependencies.find_case_result_payload
        compare_payloads_fn = dependencies.compare_result_payload_objects
        try:
            ref_payload = _resolve_case_reference_payload(
                test_file,
                ref_result,
                ref_operator_file,
                run_local_payload_fn,
                run_remote_payload_fn,
                load_case_payload_fn,
                find_case_payload_fn,
                remote,
                remote_workdir,
                case_id=case_id,
                accuracy_mode=accuracy_mode,
                keep_remote_workdir=bool(args.keep_remote_workdir),
                verbose=bool(args.verbose),
                stderr=sys.stderr,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        append_optimize_timing_event(
            timing_context,
            event="run_test_start",
            command=args.command,
            test_file=test_file,
            operator_file=operator_file,
        )
        try:
            result, candidate_payload, remote_workspace = _run_case_payload_once(
                test_file,
                operator_file,
                run_local_payload_fn,
                run_remote_payload_fn,
                remote,
                remote_workdir,
                case_id=case_id,
                accuracy_mode=accuracy_mode,
                keep_remote_workdir=bool(args.keep_remote_workdir),
                verbose=bool(args.verbose),
                stderr=sys.stderr,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            append_optimize_timing_event(
                timing_context,
                event="run_test_end",
                command=args.command,
                return_code=1,
                test_file=test_file,
                operator_file=operator_file,
            )
            print(str(exc), file=sys.stderr)
            return 1
        _render_ref_run_result(
            result,
            archived_result=None,
            remote_workspace=remote_workspace if args.keep_remote_workdir else None,
            skip_stdout=remote is not None,
        )
        final_code = int(result["return_code"])
        if final_code == 0 and ref_payload is not None:
            if candidate_payload is None:
                print(
                    "Differential run-test single-case execution did not produce a result payload required for automatic comparison.",
                    file=sys.stderr,
                )
                final_code = 1
            else:
                final_code = compare_payloads_fn(
                    ref_payload, candidate_payload, accuracy_mode=accuracy_mode
                )
        append_optimize_timing_event(
            timing_context,
            event="run_test_end",
            command=args.command,
            return_code=final_code,
            test_file=test_file,
            operator_file=operator_file,
        )
        return final_code

    ref_result = _resolve_comparison_inputs(
        parser,
        args.command,
        resolved_test_mode,
        ref_result,
        ref_operator_file,
        test_file,
        run_local_test_fn,
        run_remote_test_fn,
        remote,
        remote_workdir,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        keep_remote_workdir=bool(args.keep_remote_workdir),
        verbose=bool(args.verbose),
        require_reference_input=require_reference_input,
    )
    append_optimize_timing_event(
        timing_context,
        event="run_test_start",
        command=args.command,
        test_file=test_file,
        operator_file=operator_file,
    )
    try:
        if remote is not None:
            result, archived_result, remote_workspace = run_remote_test_fn(
                test_file,
                operator_file,
                resolved_test_mode,
                remote,
                remote_workdir,
                case_id=case_id,
                accuracy_mode=accuracy_mode,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        else:
            result, archived_result = run_local_test_fn(
                test_file,
                operator_file,
                resolved_test_mode,
                case_id=case_id,
                accuracy_mode=accuracy_mode,
                verbose=args.verbose,
            )
            remote_workspace = None
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        append_optimize_timing_event(
            timing_context,
            event="run_test_end",
            command=args.command,
            return_code=1,
            test_file=test_file,
            operator_file=operator_file,
        )
        print(str(exc), file=sys.stderr)
        return 1
    _render_result(result, skip_stdout=remote is not None)
    print(f"Return code: {result['return_code']}")
    final_code = int(result["return_code"])
    if archived_result is not None:
        print(f"Archived result: {archived_result}")
        if ref_result is not None:
            final_code = _compare_result(
                ref_result,
                archived_result,
                remote,
                remote_workdir,
                accuracy_mode=accuracy_mode,
                verbose=bool(args.verbose),
                stderr=sys.stderr,
                compare_result_fn=dependencies.compare_result_files,
                compare_remote_result_fn=dependencies.compare_remote_result_files,
            )
        if args.command == "run-test-optimize":
            cleanup_run_test_pt_files((archived_result,))
    elif ref_result is not None:
        print(
            "Differential run-test did not produce an archived result required for automatic comparison.",
            file=sys.stderr,
        )
        final_code = 1
    if remote is not None and args.keep_remote_workdir:
        print(f"Remote workspace: {remote_workspace}")
    append_optimize_timing_event(
        timing_context,
        event="run_test_end",
        command=args.command,
        return_code=final_code,
        test_file=test_file,
        operator_file=operator_file,
    )
    return final_code


def _resolve_existing_path(parser: argparse.ArgumentParser, raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        parser.error(f"{label} path does not exist: {path}")
    return path


def _resolve_optional_existing_path(
    parser: argparse.ArgumentParser, raw_path: str | None, label: str
) -> Path | None:
    return None if raw_path is None else _resolve_existing_path(parser, raw_path, label)


def _resolve_test_mode_from_metadata(test_file: Path, parse_test_metadata_fn: Any) -> str:
    mode = cast(dict[str, str], parse_test_metadata_fn(test_file)).get("test-mode")
    if mode not in {"standalone", "differential"}:
        raise ValueError(f"Test metadata is missing required 'test-mode' entry: {test_file}")
    return mode


def _validate_run_test_comparison_inputs(
    parser: argparse.ArgumentParser,
    command_name: str,
    resolved_test_mode: str,
    ref_result: Path | None,
    ref_operator_file: Path | None,
    *,
    case_id: str | None,
    require_reference_input: bool,
) -> None:
    if case_id is not None and resolved_test_mode != "differential":
        parser.error(f"{command_name} standalone mode does not accept --case-id")
    if ref_result is not None and resolved_test_mode != "differential":
        parser.error(f"{command_name} standalone mode does not accept --ref-result")
    if ref_operator_file is not None and resolved_test_mode != "differential":
        parser.error(f"{command_name} standalone mode does not accept --ref-operator-file")
    if ref_result is not None and ref_operator_file is not None:
        qualifier = "requires exactly one of" if require_reference_input else "accepts at most one of"
        parser.error(
            f"{command_name} differential mode {qualifier} --ref-result or --ref-operator-file"
        )
    if require_reference_input and resolved_test_mode == "differential" and ref_result is None and ref_operator_file is None:
        parser.error(
            f"{command_name} differential mode requires exactly one of --ref-result or --ref-operator-file"
        )


def _validate_remote_differential_inputs(
    parser: argparse.ArgumentParser,
    ref_result: Path | None,
    ref_operator_file: Path | None,
) -> None:
    if ref_result is not None:
        parser.error("Remote differential run-test does not accept --ref-result; use --ref-operator-file.")
    if ref_operator_file is None:
        parser.error("Remote differential run-test requires --ref-operator-file.")


def _resolve_comparison_inputs(
    parser: argparse.ArgumentParser,
    command_name: str,
    resolved_test_mode: str,
    ref_result: Path | None,
    ref_operator_file: Path | None,
    test_file: Path,
    run_local_test_fn: Any,
    run_remote_test_fn: Any,
    remote: str | None,
    remote_workdir: str | None,
    *,
    case_id: str | None,
    accuracy_mode: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
    require_reference_input: bool,
) -> Path | None:
    _validate_run_test_comparison_inputs(
        parser,
        command_name,
        resolved_test_mode,
        ref_result,
        ref_operator_file,
        case_id=case_id,
        require_reference_input=require_reference_input,
    )
    if ref_operator_file is None:
        return ref_result
    return _resolve_ref_operator_result(
        test_file,
        ref_operator_file,
        resolved_test_mode,
        run_local_test_fn,
        run_remote_test_fn,
        remote,
        remote_workdir,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
    )


def _resolve_ref_operator_result(
    test_file: Path,
    ref_operator_file: Path,
    test_mode: str,
    run_local_test_fn: Any,
    run_remote_test_fn: Any,
    remote: str | None,
    remote_workdir: str | None,
    *,
    case_id: str | None,
    accuracy_mode: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
) -> Path:
    derived_ref_result = _derived_result_path(ref_operator_file)
    if derived_ref_result.exists():
        return derived_ref_result
    if remote is not None:
        try:
            result, archived_result, remote_workspace = run_remote_test_fn(
                test_file,
                ref_operator_file,
                test_mode,
                remote,
                remote_workdir,
                case_id=case_id,
                accuracy_mode=accuracy_mode,
                keep_remote_workdir=keep_remote_workdir,
                verbose=verbose,
                stderr=sys.stderr,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        _render_ref_run_result(
            result,
            archived_result,
            remote_workspace=remote_workspace if keep_remote_workdir else None,
            skip_stdout=True,
        )
    else:
        try:
            result, archived_result = run_local_test_fn(
                test_file,
                ref_operator_file,
                test_mode,
                case_id=case_id,
                accuracy_mode=accuracy_mode,
                verbose=verbose,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        _render_ref_run_result(result, archived_result, remote_workspace=None)
    if int(result["return_code"]) != 0 or archived_result is None:
        raise SystemExit(1)
    return derived_ref_result


def _resolve_case_reference_payload(
    test_file: Path,
    ref_result: Path | None,
    ref_operator_file: Path | None,
    run_local_payload_fn: Any,
    run_remote_payload_fn: Any,
    load_case_payload_fn: Any,
    find_case_payload_fn: Any,
    remote: str | None,
    remote_workdir: str | None,
    *,
    case_id: str,
    accuracy_mode: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
    stderr: TextIO | None,
) -> object | None:
    if ref_result is not None:
        return load_case_payload_fn(ref_result, case_id)
    if ref_operator_file is None:
        return None
    derived_ref_result = _derived_result_path(ref_operator_file)
    if derived_ref_result.exists():
        payload = find_case_payload_fn(derived_ref_result, case_id)
        if payload is not None:
            return payload
    try:
        result, payload, remote_workspace = _run_case_payload_once(
            test_file,
            ref_operator_file,
            run_local_payload_fn,
            run_remote_payload_fn,
            remote,
            remote_workdir,
            case_id=case_id,
            accuracy_mode=accuracy_mode,
            keep_remote_workdir=keep_remote_workdir,
            verbose=verbose,
            stderr=stderr,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=stderr or sys.stderr)
        raise SystemExit(1) from exc
    _render_ref_run_result(
        result,
        archived_result=None,
        remote_workspace=remote_workspace if keep_remote_workdir else None,
        skip_stdout=remote is not None,
    )
    if int(result["return_code"]) != 0 or payload is None:
        raise SystemExit(int(result["return_code"]) or 1)
    return payload


def _run_case_payload_once(
    test_file: Path,
    operator_file: Path,
    run_local_payload_fn: Any,
    run_remote_payload_fn: Any,
    remote: str | None,
    remote_workdir: str | None,
    *,
    case_id: str,
    accuracy_mode: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
    stderr: TextIO | None,
) -> tuple[ResultPayload, object | None, str | None]:
    if remote is not None:
        result, payload, workspace = run_remote_payload_fn(
            test_file,
            operator_file,
            remote,
            remote_workdir,
            case_id=case_id,
            accuracy_mode=accuracy_mode,
            keep_remote_workdir=keep_remote_workdir,
            verbose=verbose,
            stderr=stderr,
        )
        return result, payload, workspace
    result, payload = run_local_payload_fn(
        test_file,
        operator_file,
        case_id=case_id,
        accuracy_mode=accuracy_mode,
        verbose=verbose,
    )
    return result, payload, None


def _compare_result(
    ref_result: Path,
    archived_result: Path,
    remote: str | None,
    remote_workdir: str | None,
    *,
    accuracy_mode: str | None,
    verbose: bool,
    stderr: TextIO | None,
    compare_result_fn: Any,
    compare_remote_result_fn: Any,
) -> int:
    if remote is None:
        return compare_result_fn(ref_result, archived_result, accuracy_mode=accuracy_mode)
    try:
        return compare_remote_result_fn(
            ref_result,
            archived_result,
            remote,
            remote_workdir,
            accuracy_mode=accuracy_mode,
            verbose=verbose,
            stderr=stderr,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=stderr or sys.stderr)
        return 1


def _derived_result_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def _render_ref_run_result(
    result: ResultPayload,
    archived_result: Path | None,
    *,
    remote_workspace: str | None,
    skip_stdout: bool = False,
) -> None:
    _render_result(result, skip_stdout=skip_stdout)
    print(f"Return code: {result['return_code']}")
    if archived_result is not None:
        print(f"Archived result: {archived_result}")
    if remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")


def _render_result(result: ResultPayload, skip_stdout: bool) -> None:
    stdout = result["stdout"]
    stderr = result["stderr"]
    if stdout and not skip_stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")


def _default_dependencies() -> RunTestDependencies:
    return RunTestDependencies(
        parse_test_metadata=parse_test_metadata,
        run_local_test=run_local_test,
        run_remote_test=run_remote_test,
        run_remote_differential_comparison=run_remote_differential_comparison,
        run_local_test_case_payload=run_local_test_case_payload,
        run_remote_test_case_payload=run_remote_test_case_payload,
        load_case_result_payload=load_case_result_payload,
        find_case_result_payload=find_case_result_payload,
        compare_result_payload_objects=compare_result_payload_objects,
        compare_result_files=compare_result_files,
        compare_remote_result_files=compare_remote_result_files,
    )
