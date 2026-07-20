"""Fixed remote worker protocol for run-test execution."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from result_payload import ResultPayload
from run_test_execution import run_differential_test, run_differential_test_payload, run_standalone_test
from run_test_result import differential_archive_path
from test_contract import serialize_payload_object
from torch_npu_warnings import suppress_torch_npu_owner_mismatch_warning


# Mirrored by run_test_remote_api.py; this copied remote worker must stay self-contained.
_SERIALIZED_PAYLOAD_BEGIN = "__HELIX_SERIALIZED_PAYLOAD_BEGIN__"
_SERIALIZED_PAYLOAD_END = "__HELIX_SERIALIZED_PAYLOAD_END__"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--test-mode", choices=["standalone", "differential"], required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--emit-serialized-payload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = str(Path.cwd().resolve())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    test_file = Path(args.test_file).resolve()
    operator_file = Path(args.operator_file).resolve()
    if args.test_mode == "standalone":
        if args.case_id is not None:
            raise ValueError("--case-id is supported only with differential tests.")
        if args.no_archive or args.emit_serialized_payload:
            raise ValueError("Standalone tests do not support differential payload options.")
        return _emit_result(run_standalone_test(test_file, operator_file))
    if args.emit_serialized_payload or args.no_archive:
        result, payload = run_differential_test_payload(test_file, operator_file, case_id=args.case_id)
        status = _emit_result(result)
        if args.emit_serialized_payload and status == 0 and payload is not None:
            print(_SERIALIZED_PAYLOAD_BEGIN)
            print(serialize_payload_object(payload))
            print(_SERIALIZED_PAYLOAD_END)
        return status
    archive_path = differential_archive_path(operator_file)
    result = run_differential_test(
        test_file,
        operator_file,
        archive_path,
        case_id=args.case_id,
    )
    return _emit_result(result)


def _emit_result(result: ResultPayload) -> int:
    stdout = str(result["stdout"])
    stderr = str(result["stderr"])
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
    return int(result["return_code"])


if __name__ == "__main__":
    suppress_torch_npu_owner_mismatch_warning()
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
