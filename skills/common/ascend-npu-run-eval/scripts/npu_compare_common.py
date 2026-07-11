from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
import os
from typing import cast

import torch

from env_registry import (
    HELIX_ACCURACY_MODE,
    HELIX_DTYPE_CLOSE_ATOL,
    HELIX_DTYPE_CLOSE_RTOL,
)


DEFAULT_ACCURACY_MODE = "npu-contract"
ACCURACY_MODES = frozenset({DEFAULT_ACCURACY_MODE, "dtype-close"})
_CURRENT_ACCURACY_MODE: ContextVar[str | None] = ContextVar(
    "helix_current_accuracy_mode",
    default=None,
)
FLOAT_FAMILY = {
    "float8_e4m3",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float16",
    "bfloat16",
    "float32",
    "float64",
}
COMPLEX_FAMILY = {"complex64", "complex128"}
INPUT_FLOAT_FAMILY = FLOAT_FAMILY | COMPLEX_FAMILY
OUTPUT_FLOAT_FAMILY = FLOAT_FAMILY | COMPLEX_FAMILY
INTEGER_FAMILY = {"bool", "uint8", "int8", "int16", "int32", "int64"}
DTYPE_PRIORITY = {
    "complex128": 0,
    "complex64": 1,
    "float64": 2,
    "float32": 3,
    "float16": 4,
    "bfloat16": 5,
    "float8_e4m3": 6,
    "float8_e4m3fn": 6,
    "float8_e4m3fnuz": 6,
    "float8_e5m2": 7,
    "float8_e5m2fnuz": 7,
    "int64": 8,
    "int32": 9,
    "int16": 10,
    "int8": 11,
    "uint8": 12,
    "bool": 13,
}
MATCH_THRESHOLDS = {
    "float16": {
        "small_value_threshold": 2 ** -11,
        "small_value_error": 2 ** -16,
        "rel_threshold": 2 ** -10,
    },
    "bfloat16": {
        "small_value_threshold": 2 ** -8,
        "small_value_error": 2 ** -16,
        "rel_threshold": 2 ** -7,
    },
    "float32": {
        "small_value_threshold": 2 ** -14,
        "small_value_error": 2 ** -30,
        "rel_threshold": 2 ** -13,
    },
    "hifloat32": {
        "small_value_threshold": 2 ** -12,
        "small_value_error": 2 ** -28,
        "rel_threshold": 2 ** -11,
    },
    "float8_e4m3": {
        "small_value_threshold": 2 ** -4,
        "small_value_error": 2 ** -6,
        "rel_threshold": 2 ** -3,
    },
    "float8_e5m2": {
        "small_value_threshold": 2 ** -3,
        "small_value_error": 2 ** -5,
        "rel_threshold": 2 ** -2,
    },
    "fallback": {
        "small_value_threshold": 2 ** -14,
        "small_value_error": 2 ** -30,
        "rel_threshold": 2 ** -13,
    },
}
MAX_ERROR_THRESHOLDS = {
    "float16": {"atol": 9e-2, "rtol": 2 ** -10},
    "bfloat16": {"atol": 1e-1, "rtol": 2 ** -7},
    "float32": {"atol": 1e-3, "rtol": 2 ** -13},
    "hifloat32": {"atol": 1e-3, "rtol": 2 ** -13},
    "float8_e4m3": {"atol": 1e-3, "rtol": 2 ** -13},
    "float8_e5m2": {"atol": 1e-3, "rtol": 2 ** -13},
    "fallback": {"atol": 1e-3, "rtol": 2 ** -13},
}
DTYPE_CLOSE_TOLERANCES = {
    "float64": {"rtol": 1e-5, "atol": 1e-8},
    "complex128": {"rtol": 1e-5, "atol": 1e-8},
    "float32": {"rtol": 1e-4, "atol": 1e-5},
    "hifloat32": {"rtol": 1e-4, "atol": 1e-5},
    "complex64": {"rtol": 1e-4, "atol": 1e-5},
    "float16": {"rtol": 5e-4, "atol": 5e-5},
    "bfloat16": {"rtol": 1e-3, "atol": 1e-4},
    "float8_e4m3": {"rtol": 1e-2, "atol": 1e-3},
    "float8_e5m2": {"rtol": 1e-2, "atol": 1e-3},
    "fallback": {"rtol": 1e-4, "atol": 1e-5},
}


@dataclass(frozen=True)
class CaseCompareResult:
    passed: bool
    case_id: str
    compute: bool
    input_type: str
    input_dtype: str | None
    output_dtype: str | None
    comparison_path: str
    message: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class ArtifactCompareResult:
    passed: bool
    failed_case_count: int
    case_results: tuple[CaseCompareResult, ...]
    message: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class InputInfo:
    input_type: str
    input_dtype: str | None


def resolve_accuracy_mode(accuracy_mode: str | None) -> str:
    raw_value = accuracy_mode if accuracy_mode is not None else os.environ.get(HELIX_ACCURACY_MODE)
    if raw_value is None or not raw_value.strip():
        return DEFAULT_ACCURACY_MODE
    normalized = raw_value.strip().lower()
    if normalized not in ACCURACY_MODES:
        supported = ", ".join(sorted(ACCURACY_MODES))
        raise ValueError(f"accuracy_mode must be one of {supported}, got {raw_value!r}")
    return normalized


def set_current_accuracy_mode(accuracy_mode: str) -> Token[str | None]:
    return _CURRENT_ACCURACY_MODE.set(accuracy_mode)


def reset_current_accuracy_mode(token: Token[str | None]) -> None:
    _CURRENT_ACCURACY_MODE.reset(token)


def current_accuracy_mode() -> str:
    current = _CURRENT_ACCURACY_MODE.get()
    if current is not None:
        return current
    return resolve_accuracy_mode(None)


def optional_env_float(name: str) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {raw_value!r}")
    return value


def dtype_close_tolerances_for_output_dtype(output_dtype: str) -> dict[str, float]:
    matched_key = _dtype_close_tolerance_key(output_dtype)
    matched = DTYPE_CLOSE_TOLERANCES.get(matched_key, DTYPE_CLOSE_TOLERANCES["fallback"])
    tolerances = {"atol": float(matched["atol"]), "rtol": float(matched["rtol"])}
    atol = optional_env_float(HELIX_DTYPE_CLOSE_ATOL)
    rtol = optional_env_float(HELIX_DTYPE_CLOSE_RTOL)
    if atol is not None:
        tolerances["atol"] = atol
    if rtol is not None:
        tolerances["rtol"] = rtol
    return tolerances


def thresholds_for_output_dtype(output_dtype: str) -> dict[str, float]:
    matched_key = _threshold_key(output_dtype)
    matched = MATCH_THRESHOLDS.get(matched_key, MATCH_THRESHOLDS["fallback"])
    max_error = MAX_ERROR_THRESHOLDS.get(matched_key, MAX_ERROR_THRESHOLDS["fallback"])
    return {
        "small_value_threshold": float(matched["small_value_threshold"]),
        "small_value_error": float(matched["small_value_error"]),
        "rel_threshold": float(matched["rel_threshold"]),
        "atol": float(max_error["atol"]),
        "rtol": float(max_error["rtol"]),
    }


def direct_input_values(inputs: object) -> list[object]:
    if isinstance(inputs, Mapping):
        return list(cast(Mapping[object, object], inputs).values())
    if isinstance(inputs, tuple):
        return list(cast(tuple[object, ...], inputs))
    if isinstance(inputs, list):
        return list(cast(list[object], inputs))
    if inputs is None:
        return []
    return [inputs]


def infer_input_info(inputs: object) -> InputInfo:
    direct_values = direct_input_values(inputs)
    direct_tensors = [value for value in direct_values if isinstance(value, torch.Tensor)]
    if direct_tensors:
        dtype_name = min(
            (_dtype_name(tensor.dtype) for tensor in direct_tensors),
            key=lambda name: DTYPE_PRIORITY.get(name, len(DTYPE_PRIORITY)),
        )
        return InputInfo(
            input_type="float" if dtype_name in INPUT_FLOAT_FAMILY else "int",
            input_dtype=dtype_name,
        )
    for value in direct_values:
        if isinstance(value, (list, tuple)):
            sequence_values = cast(list[object] | tuple[object, ...], value)
            tensor_items = [item for item in sequence_values if isinstance(item, torch.Tensor)]
            if tensor_items and len(tensor_items) == len(sequence_values):
                first_tensor = tensor_items[0]
                dtype_name = _dtype_name(first_tensor.dtype)
                return InputInfo(
                    input_type="float" if dtype_name in INPUT_FLOAT_FAMILY else "int",
                    input_dtype=dtype_name,
                )
    return InputInfo(input_type="no_tensor", input_dtype=None)


def coerce_output_leaf(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, bool):
        return torch.tensor(value, dtype=torch.bool)
    if isinstance(value, int):
        return torch.tensor(value, dtype=torch.int64)
    if isinstance(value, float):
        return torch.tensor(value, dtype=torch.float64)
    if isinstance(value, complex):
        return torch.tensor(value, dtype=torch.complex64)
    return None


def nan_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() or tensor.is_complex():
        return torch.isnan(tensor)
    return torch.zeros_like(tensor, dtype=torch.bool)


def inf_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() or tensor.is_complex():
        return torch.isinf(tensor)
    return torch.zeros_like(tensor, dtype=torch.bool)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def is_sequence_output(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def indices_tensor_to_tuple(indices: torch.Tensor) -> tuple[int, ...]:
    flat_indices = indices.reshape(-1)
    return tuple(int(flat_indices[index].item()) for index in range(flat_indices.numel()))


def unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...] | tuple[()]:
    if not shape:
        return ()
    if len(shape) == 1:
        return (flat_index,)
    values: list[int] = []
    remaining = flat_index
    for size in reversed(shape):
        values.append(remaining % size)
        remaining //= size
    return tuple(reversed(values))


def shape_mismatch_result(
    *,
    case_id: str,
    compute: bool,
    input_info: InputInfo,
    output_path: str,
    output_dtype: str,
    comparison_path: str,
    actual_shape: tuple[int, ...],
    golden_shape: tuple[int, ...],
) -> CaseCompareResult:
    return case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=(
            f"FAIL case '{case_id}' shape mismatch at {output_path}: "
            f"expected {golden_shape}, got {actual_shape}."
        ),
        diagnostics={
            "failure_stage": "shape_mismatch",
            "output_path": output_path,
            "shape_expected": golden_shape,
            "shape_actual": actual_shape,
            "output_dtype": output_dtype,
        },
    )


def case_result(
    *,
    passed: bool,
    case_id: str,
    compute: bool,
    input_info: InputInfo,
    output_dtype: str | None,
    comparison_path: str,
    message: str,
    diagnostics: Mapping[str, object],
) -> CaseCompareResult:
    return CaseCompareResult(
        passed=passed,
        case_id=case_id,
        compute=compute,
        input_type=input_info.input_type,
        input_dtype=input_info.input_dtype,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=message,
        diagnostics={
            "case_id": case_id,
            "compute": compute,
            "input_type": input_info.input_type,
            "input_dtype": input_info.input_dtype,
            **diagnostics,
            "accuracy_mode": current_accuracy_mode(),
        },
    )


def select_comparison_path(
    *,
    compute: bool,
    input_type: str,
    output_dtype: str,
) -> str | None:
    if not compute:
        return "non-compute"
    if output_dtype == "bool":
        return "bool-output"
    if output_dtype in INTEGER_FAMILY:
        if input_type == "float":
            return "quantized-fp-to-int"
        if input_type in {"int", "no_tensor"}:
            return "integer-compute"
        return None
    if output_dtype in OUTPUT_FLOAT_FAMILY:
        return "floating-point-compute"
    return None


def max_abs_diff(actual_cpu: torch.Tensor, golden_cpu: torch.Tensor) -> float:
    if actual_cpu.numel() == 0:
        return 0.0
    try:
        diff = (actual_cpu - golden_cpu).abs()
    except RuntimeError:
        diff = (actual_cpu.to(dtype=torch.float32) - golden_cpu.to(dtype=torch.float32)).abs()
    return float(diff.max().item())


def threshold_key(output_dtype: str) -> str:
    if output_dtype.startswith("float8_e4m3"):
        return "float8_e4m3"
    if output_dtype.startswith("float8_e5m2"):
        return "float8_e5m2"
    if output_dtype in {"float16", "bfloat16", "float32", "hifloat32"}:
        return output_dtype
    return "fallback"


def _threshold_key(output_dtype: str) -> str:
    return threshold_key(output_dtype)


def _dtype_close_tolerance_key(output_dtype: str) -> str:
    if output_dtype.startswith("float8_e4m3"):
        return "float8_e4m3"
    if output_dtype.startswith("float8_e5m2"):
        return "float8_e5m2"
    if output_dtype in {
        "float16",
        "bfloat16",
        "float32",
        "hifloat32",
        "float64",
        "complex64",
        "complex128",
    }:
        return output_dtype
    return "fallback"


def _dtype_name(dtype: torch.dtype) -> str:
    return dtype_name(dtype)
