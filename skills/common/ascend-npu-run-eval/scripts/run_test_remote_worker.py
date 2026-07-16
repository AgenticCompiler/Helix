from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

from env_registry import TRITON_ALWAYS_COMPILE
from test_contract import (
    SCRIPT_DIR,
    bootstrap_torch_npu,
    compute_flag_from_metadata,
    load_differential_test_cases,
    load_module,
    parse_test_metadata,
    require_callable,
    resolve_operator_api,
    serialize_payload_object,
    temporary_sys_path_entries,
)
from torch_npu_warnings import suppress_torch_npu_owner_mismatch_warning


# Mirrored by run_remote_api.py; this copied remote worker must stay self-contained.
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


def _synchronize(torch_module: object) -> None:
    npu = getattr(torch_module, "npu", None)
    synchronize = getattr(npu, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _run_standalone(test_file: Path, operator_file: Path) -> None:
    metadata = parse_test_metadata(test_file)
    bootstrap_torch_npu(test_file.parent, operator_file.parent)
    with temporary_sys_path_entries(test_file.parent, operator_file.parent, SCRIPT_DIR):
        test_module = load_module(test_file, f"standalone_test_{test_file.stem}")
        main_fn = require_callable(test_module, "main", test_file, kind="Standalone test module")
        operator_module = load_module(operator_file, f"standalone_operator_{operator_file.stem}")
        main_fn(resolve_operator_api(operator_module, metadata, operator_file))
        torch = importlib.import_module("torch")
        _synchronize(torch)


def _run_differential(
    test_file: Path,
    operator_file: Path,
    *,
    case_id: str | None,
    archive: bool,
    emit_serialized_payload: bool,
) -> None:
    bootstrap_torch_npu(test_file.parent, operator_file.parent)
    torch = importlib.import_module("torch")
    compute = compute_flag_from_metadata(parse_test_metadata(test_file))
    records: list[dict[str, object]] = []
    for case in load_differential_test_cases(test_file, operator_file, case_id=case_id):
        records.append({"id": case.case_id, "inputs": case.inputs, "result": case.fn()})
        _synchronize(torch)
    payload = {"compute": compute, "cases": records}
    if archive:
        torch.save(payload, operator_file.parent / f"{operator_file.stem}_result.pt")
    if emit_serialized_payload:
        print(_SERIALIZED_PAYLOAD_BEGIN)
        print(serialize_payload_object(payload))
        print(_SERIALIZED_PAYLOAD_END)


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
        _run_standalone(test_file, operator_file)
        return 0
    _run_differential(
        test_file,
        operator_file,
        case_id=args.case_id,
        archive=not args.no_archive,
        emit_serialized_payload=args.emit_serialized_payload,
    )
    return 0


if __name__ == "__main__":
    suppress_torch_npu_owner_mismatch_warning()
    previous = os.environ.get(TRITON_ALWAYS_COMPILE)
    os.environ[TRITON_ALWAYS_COMPILE] = "1"
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        if previous is None:
            os.environ.pop(TRITON_ALWAYS_COMPILE, None)
        else:
            os.environ[TRITON_ALWAYS_COMPILE] = previous
