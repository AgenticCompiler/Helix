from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, TextIO, cast

from env_registry import (
    HELIX_ACCURACY_MODE,
    HELIX_DTYPE_CLOSE_ATOL,
    HELIX_DTYPE_CLOSE_RTOL,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-result", "--oracle-result", dest="ref_result", required=True)
    parser.add_argument("--new-result", required=True)
    parser.add_argument(
        "--accuracy-mode",
        choices=["npu-contract", "dtype-close"],
        default=None,
    )
    args = parser.parse_args()
    return compare_result_files(args.ref_result, args.new_result, accuracy_mode=args.accuracy_mode)


@lru_cache(maxsize=1)
def _load_npu_compare_module() -> ModuleType:
    return _load_sibling_module("npu_compare.py", "compare_result_npu_compare")


@lru_cache(maxsize=1)
def _load_run_runtime_module() -> ModuleType:
    return _load_sibling_module("run_runtime.py", "compare_result_run_runtime")


def _load_sibling_module(filename: str, module_name: str) -> ModuleType:
    module_path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    scripts_root = str(module_path.parent)
    added = False
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
        added = True
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
        if added:
            sys.path.remove(scripts_root)
    return module


def compare_result_payload_objects(
    ref_payload: object,
    new_payload: object,
    *,
    accuracy_mode: str | None = None,
) -> int:
    npu_compare = _load_npu_compare_module()
    compare_result_payloads = getattr(npu_compare, "compare_result_payloads")
    format_artifact_compare_result = getattr(npu_compare, "format_artifact_compare_result")
    result = compare_result_payloads(
        ref_payload,
        new_payload,
        accuracy_mode=accuracy_mode,
    )
    print(format_artifact_compare_result(result))
    return 0 if result.passed else 1


def compare_result_files(
    ref_result: str | Path,
    new_result: str | Path,
    *,
    accuracy_mode: str | None = None,
) -> int:
    return compare_result_payload_objects(
        load_result_payload(ref_result),
        load_result_payload(new_result),
        accuracy_mode=accuracy_mode,
    )


def compare_remote_result_files(
    ref_result: Path,
    new_result: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    accuracy_mode: str | None = None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    run_runtime = _load_run_runtime_module()
    cleanup_remote_workspace = getattr(run_runtime, "cleanup_remote_workspace")
    copy_file_to_remote = getattr(run_runtime, "copy_file_to_remote")
    copy_npu_compare_runtime_to_remote = getattr(run_runtime, "copy_npu_compare_runtime_to_remote")
    create_remote_workspace = getattr(run_runtime, "create_remote_workspace")
    run_remote_command_streaming = getattr(run_runtime, "run_remote_command_streaming")

    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    compare_script = Path(__file__).resolve()
    remote_script = f"{remote_workspace}/{compare_script.name}"
    remote_ref = f"{remote_workspace}/{ref_result.name}"
    remote_new = f"{remote_workspace}/{new_result.name}"
    try:
        copy_file_to_remote(spec, compare_script, remote_script, verbose=verbose, stderr=stderr)
        copy_npu_compare_runtime_to_remote(
            spec,
            compare_script.parent,
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
            copy_file_fn=copy_file_to_remote,
        )
        copy_file_to_remote(spec, ref_result, remote_ref, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, new_result, remote_new, verbose=verbose, stderr=stderr)
        command = [
            "python3",
            compare_script.name,
            "--ref-result",
            ref_result.name,
            "--new-result",
            new_result.name,
        ]
        if accuracy_mode is not None:
            command.extend(["--accuracy-mode", accuracy_mode])
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            command,
            verbose=verbose,
            stderr=stderr,
            extra_env=_comparison_extra_env(accuracy_mode),
        )
        return int(result["return_code"])
    finally:
        cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _comparison_extra_env(accuracy_mode: str | None = None) -> dict[str, str]:
    extra_env: dict[str, str] = {}
    if accuracy_mode is not None:
        extra_env[HELIX_ACCURACY_MODE] = accuracy_mode
    for name in (
        HELIX_ACCURACY_MODE,
        HELIX_DTYPE_CLOSE_ATOL,
        HELIX_DTYPE_CLOSE_RTOL,
    ):
        if name in extra_env:
            continue
        value = os.environ.get(name)
        if value is not None:
            extra_env[name] = value
    return extra_env


def load_result_payload(path: str | Path) -> object:
    torch_module = cast(Any, importlib.import_module("torch"))
    return torch_module.load(Path(path), map_location="cpu")


def find_case_result_payload(path: str | Path, case_id: str) -> object | None:
    return _find_case_result_payload(load_result_payload(path), case_id, label=str(Path(path)))


def load_case_result_payload(path: str | Path, case_id: str) -> object:
    payload = find_case_result_payload(path, case_id)
    if payload is not None:
        return payload
    available = ", ".join(_list_result_case_ids(load_result_payload(path), label=str(Path(path))))
    raise ValueError(
        f"Result payload '{Path(path)}' does not contain case '{case_id}'. Available case ids: {available}"
    )


def _find_case_result_payload(
    payload: object,
    case_id: str,
    *,
    label: str,
) -> object | None:
    compute, cases = _extract_payload_cases(payload, label=label)
    for case in cases:
        if case["id"] == case_id:
            return {
                "compute": compute,
                "cases": [
                    {
                        "id": case["id"],
                        "inputs": case["inputs"],
                        "result": case["result"],
                    }
                ],
            }
    return None


def _list_result_case_ids(payload: object, *, label: str) -> list[str]:
    _compute, cases = _extract_payload_cases(payload, label=label)
    return [cast(str, case["id"]) for case in cases]


def _extract_payload_cases(
    payload: object,
    *,
    label: str,
) -> tuple[bool, list[dict[str, object]]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Result payload '{label}' must be a dict with a 'cases' entry.")
    payload_map = cast(Mapping[str, object], payload)
    if "results" in payload_map:
        raise ValueError(
            f"Result payload '{label}' uses the legacy payload format. "
            "Expected {'compute': <bool>, 'cases': [...]} instead of {'results': [...]}."
        )
    raw_cases = payload_map.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"Result payload '{label}' is missing required list field 'cases'.")
    raw_compute = payload_map.get("compute")
    compute = raw_compute if isinstance(raw_compute, bool) else True
    cases: list[dict[str, object]] = []
    for raw_case in cast(list[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"Result payload '{label}' contains a non-mapping case entry.")
        case_map = cast(Mapping[str, object], raw_case)
        case_id = case_map.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Result payload '{label}' contains a case without a valid string id.")
        if "inputs" not in case_map:
            raise ValueError(f"Result payload '{label}' case '{case_id}' is missing required field 'inputs'.")
        if "result" not in case_map:
            raise ValueError(f"Result payload '{label}' case '{case_id}' is missing required field 'result'.")
        cases.append(
            {
                "id": case_id,
                "inputs": case_map["inputs"],
                "result": case_map["result"],
            }
        )
    return compute, cases


if __name__ == "__main__":
    raise SystemExit(main())
