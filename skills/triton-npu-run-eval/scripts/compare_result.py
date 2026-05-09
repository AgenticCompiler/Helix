from __future__ import annotations

import argparse
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO, cast


ORACLE_COMPARE_LEVELS = {
    "strict": (1e-5, 1e-6),
    "balanced": (1e-4, 1e-5),
    "relaxed": (1e-3, 1e-4),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-result", required=True)
    parser.add_argument("--new-result", required=True)
    parser.add_argument(
        "--compare-level",
        default="balanced",
        choices=["strict", "balanced", "relaxed"],
    )
    args = parser.parse_args()
    return compare_result_files(args.oracle_result, args.new_result, args.compare_level)


def compare_result_files(
    oracle_result: str | Path,
    new_result: str | Path,
    compare_level: str,
) -> int:
    try:
        rtol, atol = _resolve_compare_tolerances(compare_level)
    except ValueError:
        print(
            f"FAIL: invalid compare level '{compare_level}', "
            f"expected one of {sorted(ORACLE_COMPARE_LEVELS)}"
        )
        return 2

    expected_payload = _load_result_payload(oracle_result)
    actual_payload = _load_result_payload(new_result)

    expected, expected_error = _extract_ordered_results(expected_payload, "oracle")
    if expected_error:
        print(f"FAIL: {expected_error}")
        return 1

    actual, actual_error = _extract_ordered_results(actual_payload, "compare")
    if actual_error:
        print(f"FAIL: {actual_error}")
        return 1

    mismatch = _compare_values(expected, actual, "output", rtol, atol)
    if mismatch:
        print(f"FAIL: {mismatch}")
        return 1

    print(
        "PASS: ordered outputs match "
        f"(level={compare_level.strip().lower()}, rtol={rtol}, atol={atol})"
    )
    return 0


def compare_remote_result_files(
    oracle_result: Path,
    new_result: Path,
    compare_level: str,
    remote: str,
    remote_workdir: str | None,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    from run_runtime import (
        cleanup_remote_workspace,
        copy_file_to_remote,
        create_remote_workspace,
        run_remote_command_streaming,
    )

    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    compare_script = Path(__file__).resolve()
    remote_script = f"{remote_workspace}/{compare_script.name}"
    remote_oracle = f"{remote_workspace}/{oracle_result.name}"
    remote_new = f"{remote_workspace}/{new_result.name}"
    try:
        copy_file_to_remote(spec, compare_script, remote_script, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, oracle_result, remote_oracle, verbose=verbose, stderr=stderr)
        copy_file_to_remote(spec, new_result, remote_new, verbose=verbose, stderr=stderr)
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                compare_script.name,
                "--oracle-result",
                oracle_result.name,
                "--new-result",
                new_result.name,
                "--compare-level",
                compare_level,
            ],
            verbose=verbose,
            stderr=stderr,
        )
        return int(result["return_code"])
    finally:
        cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _resolve_compare_tolerances(level: str) -> tuple[float, float]:
    normalized = level.strip().lower()
    if normalized not in ORACLE_COMPARE_LEVELS:
        raise ValueError(normalized)
    return ORACLE_COMPARE_LEVELS[normalized]


def _load_result_payload(path: str | Path) -> Any:
    torch = importlib.import_module("torch")
    return torch.load(Path(path), map_location="cpu")


def _extract_ordered_results(payload: Any, label: str) -> tuple[list[object] | None, str | None]:
    if not isinstance(payload, Mapping):
        return None, f"{label} payload must be a dict with a 'results' entry"
    payload_dict = cast(Mapping[str, object], payload)
    if "results" not in payload_dict:
        return None, f"{label} payload is missing required 'results' entry"
    results = payload_dict["results"]
    if not isinstance(results, list):
        return None, f"{label} payload 'results' must be a list"
    return cast(list[object], results), None


def _compare_values(expected: Any, actual: Any, path: str, rtol: float, atol: float) -> str | None:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            return f"{path} type mismatch: expected Tensor, got {type(actual).__name__}"
        if expected.shape != actual.shape:
            return f"{path} shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
        if expected.dtype != actual.dtype:
            return f"{path} dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        expected_cpu = expected.detach().cpu()
        actual_cpu = actual.detach().cpu()
        if not torch.allclose(expected_cpu, actual_cpu, rtol=rtol, atol=atol, equal_nan=True):
            diff = (expected_cpu - actual_cpu).abs().max().item()
            return f"{path} tensor mismatch: max diff={diff}, rtol={rtol}, atol={atol}"
        return None

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return f"{path} type mismatch: expected mapping, got {type(actual).__name__}"
        expected_mapping = cast(Mapping[object, object], expected)
        actual_mapping = cast(Mapping[object, object], actual)
        if set(expected_mapping.keys()) != set(actual_mapping.keys()):
            return f"{path} key mismatch: expected {sorted(expected_mapping.keys(), key=str)}, got {sorted(actual_mapping.keys(), key=str)}"
        for key in sorted(expected_mapping.keys(), key=str):
            mismatch = _compare_values(expected_mapping[key], actual_mapping[key], f"{path}.{key}", rtol, atol)
            if mismatch:
                return mismatch
        return None

    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
            return f"{path} type mismatch: expected sequence, got {type(actual).__name__}"
        expected_seq = cast(Sequence[object], expected)
        actual_seq = cast(Sequence[object], actual)
        if len(expected_seq) != len(actual_seq):
            return f"{path} length mismatch: expected {len(expected_seq)}, got {len(actual_seq)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected_seq, actual_seq)):
            mismatch = _compare_values(expected_item, actual_item, f"{path}[{index}]", rtol, atol)
            if mismatch:
                return mismatch
        return None

    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(expected) is not type(actual) or expected != actual:
            return f"{path} value mismatch: expected {expected!r}, got {actual!r}"
        return None

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        exp_f, act_f = float(expected), float(actual)
        import math

        exp_nan, act_nan = math.isnan(exp_f), math.isnan(act_f)
        if exp_nan or act_nan:
            if exp_nan != act_nan:
                return f"{path} NaN mismatch: expected {expected}, got {actual}"
            return None
        if abs(exp_f - act_f) > (atol + rtol * abs(exp_f)):
            return f"{path} scalar mismatch: expected {expected}, got {actual}"
        return None

    if expected != actual:
        return f"{path} value mismatch: expected {expected!r}, got {actual!r}"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
