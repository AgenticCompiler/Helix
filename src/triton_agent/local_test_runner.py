from __future__ import annotations

import shutil
import sys
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from triton_agent.models import AgentResult
from triton_agent.process_runner import run_process


ORACLE_COMPARE_LEVELS = {
    "strict": (1e-5, 1e-6),
    "balanced": (1e-4, 1e-5),
    "relaxed": (1e-3, 1e-4),
}


def run_local_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
) -> tuple[AgentResult, Path | None]:
    command = [sys.executable, str(test_file), "--operator-file", str(operator_file)]
    result = run_process(command, str(test_file.parent), mode="streaming", stall_timeout_seconds=900)
    archived_result = None
    if test_mode == "differential" and result.succeeded:
        archived_result = archive_differential_result(test_file, operator_file)
    return result, archived_result


def find_case_insensitive_result_file(directory: Path) -> Path | None:
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.name.lower() == "test_result.pt":
            return candidate
    return None


def archive_differential_result(test_file: Path, operator_file: Path) -> Path:
    result_file = find_case_insensitive_result_file(operator_file.parent)
    if result_file is None and test_file.parent != operator_file.parent:
        result_file = find_case_insensitive_result_file(test_file.parent)
    if result_file is None:
        raise FileNotFoundError(
            f"Differential result payload not found beside operator or test file: {operator_file.parent}"
        )

    archive_name = f"{operator_file.stem}_result.pt"
    archive_path = operator_file.parent / archive_name
    shutil.copy2(result_file, archive_path)
    return archive_path


def compare_result_files(oracle_result: Path, new_result: Path, compare_level: str) -> int:
    return _compare_result_files_impl(oracle_result, new_result, compare_level)


def _compare_result_files_impl(oracle_result: Path, new_result: Path, compare_level: str) -> int:
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


def _resolve_compare_tolerances(level: str) -> tuple[float, float]:
    normalized = level.strip().lower()
    if normalized not in ORACLE_COMPARE_LEVELS:
        raise ValueError(normalized)
    return ORACLE_COMPARE_LEVELS[normalized]


def _load_result_payload(path: Path) -> Any:
    torch = importlib.import_module("torch")
    return torch.load(path, map_location="cpu")


def _extract_ordered_results(payload: Any, label: str) -> tuple[list[Any] | None, str | None]:
    if not isinstance(payload, Mapping):
        return None, f"{label} payload must be a dict with a 'results' entry"
    if "results" not in payload:
        return None, f"{label} payload is missing required 'results' entry"
    results = payload["results"]
    if not isinstance(results, list):
        return None, f"{label} payload 'results' must be a list"
    return results, None


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
        if set(expected.keys()) != set(actual.keys()):
            return f"{path} key mismatch: expected {sorted(expected.keys())}, got {sorted(actual.keys())}"
        for key in sorted(expected.keys(), key=str):
            mismatch = _compare_values(expected[key], actual[key], f"{path}.{key}", rtol, atol)
            if mismatch:
                return mismatch
        return None

    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
            return f"{path} type mismatch: expected sequence, got {type(actual).__name__}"
        if len(expected) != len(actual):
            return f"{path} length mismatch: expected {len(expected)}, got {len(actual)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            mismatch = _compare_values(expected_item, actual_item, f"{path}[{index}]", rtol, atol)
            if mismatch:
                return mismatch
        return None

    if isinstance(expected, float):
        if not isinstance(actual, (int, float)):
            return f"{path} type mismatch: expected float-like, got {type(actual).__name__}"
        if abs(expected - float(actual)) > (atol + rtol * abs(expected)):
            return f"{path} scalar mismatch: expected {expected}, got {actual}"
        return None

    if expected != actual:
        return f"{path} value mismatch: expected {expected!r}, got {actual!r}"
    return None
