from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch


_FLOAT_FAMILY = {
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
_COMPLEX_FAMILY = {"complex64", "complex128"}
_INPUT_FLOAT_FAMILY = _FLOAT_FAMILY | _COMPLEX_FAMILY
_OUTPUT_FLOAT_FAMILY = _FLOAT_FAMILY | _COMPLEX_FAMILY
_INTEGER_FAMILY = {"bool", "uint8", "int8", "int16", "int32", "int64"}
_DTYPE_PRIORITY = {
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
_MATCH_THRESHOLDS = {
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
_MAX_ERROR_THRESHOLDS = {
    "float16": {
        "atol": 9e-2,
        "rtol": 2 ** -10,
    },
    "bfloat16": {
        "atol": 1e-1,
        "rtol": 2 ** -7,
    },
    "float32": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "hifloat32": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "float8_e4m3": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "float8_e5m2": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
    "fallback": {
        "atol": 1e-3,
        "rtol": 2 ** -13,
    },
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
class _InputInfo:
    input_type: str
    input_dtype: str | None


def compare_case_result(
    *,
    case_id: str,
    actual: object,
    golden: object,
    inputs: object,
    compute: bool = True,
) -> CaseCompareResult:
    input_info = _infer_input_info(inputs)
    leaf_results = _compare_value(
        actual=actual,
        golden=golden,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_path="output",
    )
    if not leaf_results:
        return _case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=None,
            comparison_path="empty-output",
            message=f"PASS case '{case_id}' has no comparable outputs.",
            diagnostics={"case_id": case_id, "output_path": "output"},
        )
    failed = next((result for result in leaf_results if not result.passed), None)
    if failed is not None:
        return failed
    first = leaf_results[0]
    output_dtype = first.output_dtype if len({result.output_dtype for result in leaf_results}) == 1 else "multiple"
    comparison_path = first.comparison_path if len({result.comparison_path for result in leaf_results}) == 1 else "composite"
    return _case_result(
        passed=True,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=f"PASS case '{case_id}' matched across {len(leaf_results)} output leaf/leaves.",
        diagnostics={
            "case_id": case_id,
            "compute": compute,
            "input_type": input_info.input_type,
            "input_dtype": input_info.input_dtype,
            "output_path": "output",
            "leaf_count": len(leaf_results),
        },
    )


def compare_result_payloads(
    oracle_payload: object,
    actual_payload: object,
) -> ArtifactCompareResult:
    oracle_cases, oracle_error = _extract_case_records(oracle_payload, "oracle")
    if oracle_error is not None:
        return ArtifactCompareResult(
            passed=False,
            failed_case_count=1,
            case_results=(),
            message=f"FAIL: {oracle_error}",
            diagnostics={"payload": "oracle"},
        )
    actual_cases, actual_error = _extract_case_records(actual_payload, "actual_payload")
    if actual_error is not None:
        return ArtifactCompareResult(
            passed=False,
            failed_case_count=1,
            case_results=(),
            message=f"FAIL: {actual_error}",
            diagnostics={"payload": "actual_payload"},
        )
    if len(oracle_cases) != len(actual_cases):
        return ArtifactCompareResult(
            passed=False,
            failed_case_count=1,
            case_results=(),
            message=(
                "FAIL: payload case count mismatch: "
                f"oracle={len(oracle_cases)}, actual_payload={len(actual_cases)}"
            ),
            diagnostics={"oracle_case_count": len(oracle_cases), "actual_case_count": len(actual_cases)},
        )
    oracle_compute = _payload_compute_flag(oracle_payload)
    actual_compute = _payload_compute_flag(actual_payload)
    if oracle_compute != actual_compute:
        return ArtifactCompareResult(
            passed=False,
            failed_case_count=1,
            case_results=(),
            message=(
                "FAIL: payload compute-kind mismatch: "
                f"oracle={oracle_compute}, actual_payload={actual_compute}"
            ),
            diagnostics={"oracle_compute": oracle_compute, "actual_compute": actual_compute},
        )
    case_results: list[CaseCompareResult] = []
    for index, (oracle_case, actual_case) in enumerate(zip(oracle_cases, actual_cases)):
        if oracle_case.case_id != actual_case.case_id:
            case_results.append(
                _case_result(
                    passed=False,
                    case_id=oracle_case.case_id,
                    compute=oracle_compute,
                    input_info=_infer_input_info(oracle_case.inputs),
                    output_dtype=None,
                    comparison_path="payload-case-order",
                    message=(
                        "FAIL payload case order mismatch at index "
                        f"{index}: oracle={oracle_case.case_id!r}, actual_payload={actual_case.case_id!r}"
                    ),
                    diagnostics={
                        "failure_stage": "case_id_mismatch",
                        "index": index,
                        "oracle_case_id": oracle_case.case_id,
                        "actual_case_id": actual_case.case_id,
                    },
                )
            )
            continue
        case_results.append(
            compare_case_result(
                case_id=oracle_case.case_id,
                actual=actual_case.result,
                golden=oracle_case.result,
                inputs=oracle_case.inputs,
                compute=oracle_compute,
            )
        )
    failed = [result for result in case_results if not result.passed]
    if failed:
        return ArtifactCompareResult(
            passed=False,
            failed_case_count=len(failed),
            case_results=tuple(case_results),
            message=(
                f"FAIL: {len(failed)} of {len(case_results)} case(s) failed. "
                f"First failure: {failed[0].message}"
            ),
            diagnostics={"case_count": len(case_results)},
        )
    return ArtifactCompareResult(
        passed=True,
        failed_case_count=0,
        case_results=tuple(case_results),
        message=f"PASS: all {len(case_results)} case(s) matched the NPU accuracy contract.",
        diagnostics={"case_count": len(case_results)},
    )


def format_artifact_compare_result(result: ArtifactCompareResult) -> str:
    lines = [result.message]
    if not result.passed:
        for case_result in result.case_results:
            if not case_result.passed:
                lines.append(case_result.message)
                diagnostics = case_result.diagnostics
                threshold_info = diagnostics.get("thresholds")
                if isinstance(threshold_info, Mapping):
                    lines.append(f"  thresholds={dict(cast(Mapping[object, object], threshold_info))}")
                lines.append(f"  diagnostics={dict(diagnostics)}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _CaseRecord:
    case_id: str
    inputs: object
    result: object


def _extract_case_records(payload: object, label: str) -> tuple[list[_CaseRecord], str | None]:
    if not isinstance(payload, Mapping):
        return [], f"{label} payload must be a dict with a 'cases' entry"
    payload_map = cast(Mapping[str, object], payload)
    if "results" in payload_map:
        return [], (
            f"{label} payload uses the legacy payload format. "
            "Expected {'compute': <bool>, 'cases': [...]} instead of {'results': [...]}."
        )
    raw_cases = payload_map.get("cases")
    if not isinstance(raw_cases, list):
        return [], f"{label} payload 'cases' must be a list"
    records: list[_CaseRecord] = []
    for raw_case in cast(list[object], raw_cases):
        if not isinstance(raw_case, Mapping):
            return [], f"{label} payload cases must be mappings"
        case_map = cast(Mapping[str, object], raw_case)
        case_id = case_map.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            return [], f"{label} payload case is missing required string field 'id'"
        if "inputs" not in case_map:
            return [], f"{label} payload case '{case_id}' is missing required field 'inputs'"
        if "result" not in case_map:
            return [], f"{label} payload case '{case_id}' is missing required field 'result'"
        records.append(
            _CaseRecord(
                case_id=case_id,
                inputs=case_map["inputs"],
                result=case_map["result"],
            )
        )
    return records, None


def _payload_compute_flag(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return True
    payload_map = cast(Mapping[str, object], payload)
    raw_compute = payload_map.get("compute")
    if raw_compute is None:
        return True
    if isinstance(raw_compute, bool):
        return raw_compute
    return True


def _compare_value(
    *,
    actual: object,
    golden: object,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
) -> list[CaseCompareResult]:
    if isinstance(golden, Mapping):
        if not isinstance(actual, Mapping):
            return [
                _case_result(
                    passed=False,
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_dtype=None,
                    comparison_path="structure-mismatch",
                    message=(
                        f"FAIL case '{case_id}' expected mapping at {output_path}, "
                        f"got {type(actual).__name__}."
                    ),
                    diagnostics={
                        "failure_stage": "type_mismatch",
                        "output_path": output_path,
                        "expected_type": "mapping",
                        "actual_type": type(actual).__name__,
                    },
                )
            ]
        golden_map = cast(Mapping[object, object], golden)
        actual_map = cast(Mapping[object, object], actual)
        golden_keys = set(golden_map.keys())
        actual_keys = set(actual_map.keys())
        if golden_keys != actual_keys:
            return [
                _case_result(
                    passed=False,
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_dtype=None,
                    comparison_path="structure-mismatch",
                    message=f"FAIL case '{case_id}' mapping keys differ at {output_path}.",
                    diagnostics={
                        "failure_stage": "key_mismatch",
                        "output_path": output_path,
                        "expected_keys": sorted(golden_keys, key=str),
                        "actual_keys": sorted(actual_keys, key=str),
                    },
                )
            ]
        results: list[CaseCompareResult] = []
        for key in sorted(golden_keys, key=lambda item: str(item)):
            child_path = f"{output_path}.{key}"
            results.extend(
                _compare_value(
                    actual=actual_map[key],
                    golden=golden_map[key],
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_path=child_path,
                )
            )
        return results
    if _is_sequence_output(golden):
        if not _is_sequence_output(actual):
            return [
                _case_result(
                    passed=False,
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_dtype=None,
                    comparison_path="structure-mismatch",
                    message=(
                        f"FAIL case '{case_id}' expected sequence at {output_path}, "
                        f"got {type(actual).__name__}."
                    ),
                    diagnostics={
                        "failure_stage": "type_mismatch",
                        "output_path": output_path,
                        "expected_type": "sequence",
                        "actual_type": type(actual).__name__,
                    },
                )
            ]
        golden_seq = list(cast(Sequence[object], golden))
        actual_seq = list(cast(Sequence[object], actual))
        if len(golden_seq) != len(actual_seq):
            return [
                _case_result(
                    passed=False,
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_dtype=None,
                    comparison_path="structure-mismatch",
                    message=f"FAIL case '{case_id}' sequence length differs at {output_path}.",
                    diagnostics={
                        "failure_stage": "length_mismatch",
                        "output_path": output_path,
                        "expected_length": len(golden_seq),
                        "actual_length": len(actual_seq),
                    },
                )
            ]
        results: list[CaseCompareResult] = []
        for index, (actual_item, golden_item) in enumerate(zip(actual_seq, golden_seq)):
            results.extend(
                _compare_value(
                    actual=actual_item,
                    golden=golden_item,
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_path=f"{output_path}[{index}]",
                )
            )
        return results
    return [
        _compare_leaf(
            actual=actual,
            golden=golden,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
        )
    ]


def _compare_leaf(
    *,
    actual: object,
    golden: object,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
) -> CaseCompareResult:
    actual_tensor = _coerce_output_leaf(actual)
    golden_tensor = _coerce_output_leaf(golden)
    if actual_tensor is None or golden_tensor is None:
        if not compute and actual == golden:
            return _case_result(
                passed=True,
                case_id=case_id,
                compute=compute,
                input_info=input_info,
                output_dtype=None,
                comparison_path="non-compute",
                message=f"PASS case '{case_id}' matched non-tensor output at {output_path}.",
                diagnostics={"output_path": output_path},
            )
        return _case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=None,
            comparison_path="unsupported-output-type",
            message=(
                f"FAIL case '{case_id}' produced unsupported output types at {output_path}: "
                f"expected {type(golden).__name__}, got {type(actual).__name__}."
            ),
            diagnostics={
                "failure_stage": "unsupported_output_type",
                "output_path": output_path,
                "expected_type": type(golden).__name__,
                "actual_type": type(actual).__name__,
            },
        )
    actual_cpu = actual_tensor.detach().cpu()
    golden_cpu = golden_tensor.detach().cpu()
    output_dtype = _dtype_name(golden_cpu.dtype)
    comparison_path = _select_comparison_path(
        compute=compute,
        input_type=input_info.input_type,
        output_dtype=output_dtype,
    )
    if comparison_path is None:
        return _case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="unsupported-output-type",
            message=f"FAIL case '{case_id}' output dtype '{output_dtype}' is unsupported at {output_path}.",
            diagnostics={
                "failure_stage": "unsupported_output_type",
                "output_path": output_path,
                "output_dtype": output_dtype,
            },
        )
    if comparison_path == "non-compute":
        return _compare_non_compute(
            actual_cpu=actual_cpu,
            golden_cpu=golden_cpu,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
        )
    precheck_failure = _run_prechecks(
        actual_cpu=actual_cpu,
        golden_cpu=golden_cpu,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_path=output_path,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
    )
    if precheck_failure is not None:
        return precheck_failure
    if comparison_path == "bool-output":
        return _compare_bool_output(
            actual_cpu=actual_cpu,
            golden_cpu=golden_cpu,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
        )
    if comparison_path == "integer-compute":
        return _compare_integer_output(
            actual_cpu=actual_cpu,
            golden_cpu=golden_cpu,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
            bound=0,
            comparison_path=comparison_path,
        )
    if comparison_path == "quantized-fp-to-int":
        return _compare_integer_output(
            actual_cpu=actual_cpu,
            golden_cpu=golden_cpu,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
            bound=1,
            comparison_path=comparison_path,
        )
    return _compare_floating_output(
        actual_cpu=actual_cpu,
        golden_cpu=golden_cpu,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_path=output_path,
        output_dtype=output_dtype,
    )


def _compare_non_compute(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
    output_dtype: str,
) -> CaseCompareResult:
    if tuple(actual_cpu.shape) != tuple(golden_cpu.shape):
        return _shape_mismatch_result(
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
            comparison_path="non-compute",
            actual_shape=tuple(actual_cpu.shape),
            golden_shape=tuple(golden_cpu.shape),
        )
    if actual_cpu.dtype != golden_cpu.dtype:
        return _case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="non-compute",
            message=f"FAIL case '{case_id}' dtype mismatch at {output_path} for non-compute comparison.",
            diagnostics={
                "failure_stage": "dtype_mismatch",
                "output_path": output_path,
                "expected_dtype": _dtype_name(golden_cpu.dtype),
                "actual_dtype": _dtype_name(actual_cpu.dtype),
            },
        )
    actual_bytes = actual_cpu.contiguous().view(torch.uint8)
    golden_bytes = golden_cpu.contiguous().view(torch.uint8)
    if torch.equal(actual_bytes, golden_bytes):
        return _case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="non-compute",
            message=f"PASS case '{case_id}' matched raw bits at {output_path}.",
            diagnostics={"output_path": output_path},
        )
    return _case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path="non-compute",
        message=f"FAIL case '{case_id}' raw bits differ at {output_path}.",
        diagnostics={
            "failure_stage": "binary_equal",
            "output_path": output_path,
            "expected_dtype": _dtype_name(golden_cpu.dtype),
            "actual_dtype": _dtype_name(actual_cpu.dtype),
        },
    )


def _run_prechecks(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
    output_dtype: str,
    comparison_path: str,
) -> CaseCompareResult | None:
    if tuple(actual_cpu.shape) != tuple(golden_cpu.shape):
        return _shape_mismatch_result(
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            actual_shape=tuple(actual_cpu.shape),
            golden_shape=tuple(golden_cpu.shape),
        )
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    nan_mask_actual = _nan_mask(actual_cast)
    nan_mask_golden = _nan_mask(golden_cpu)
    if not torch.equal(nan_mask_actual, nan_mask_golden):
        mismatch_count = int(torch.count_nonzero(nan_mask_actual != nan_mask_golden).item())
        return _case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"FAIL case '{case_id}' NaN locations differ at {output_path}.",
            diagnostics={
                "failure_stage": "nan_mask_mismatch",
                "output_path": output_path,
                "nan_mismatch_count": mismatch_count,
            },
        )
    inf_mask_actual = _inf_mask(actual_cast)
    inf_mask_golden = _inf_mask(golden_cpu)
    if not torch.equal(inf_mask_actual, inf_mask_golden):
        mismatch_count = int(torch.count_nonzero(inf_mask_actual != inf_mask_golden).item())
        return _case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"FAIL case '{case_id}' Inf locations differ at {output_path}.",
            diagnostics={
                "failure_stage": "inf_mask_mismatch",
                "output_path": output_path,
                "inf_mismatch_count": mismatch_count,
            },
        )
    has_inf = int(torch.count_nonzero(inf_mask_golden).item()) > 0
    if has_inf and not torch.equal(actual_cast[inf_mask_golden], golden_cpu[inf_mask_golden]):
        return _case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"FAIL case '{case_id}' Inf values/signs differ at {output_path}.",
            diagnostics={
                "failure_stage": "inf_sign_mismatch",
                "output_path": output_path,
                "inf_mismatch_count": int(
                    torch.count_nonzero(
                        actual_cast[inf_mask_golden] != golden_cpu[inf_mask_golden]
                    ).item()
                ),
            },
        )
    return None


def _compare_bool_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
    output_dtype: str,
) -> CaseCompareResult:
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    if torch.equal(actual_cast, golden_cpu):
        return _case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="bool-output",
            message=f"PASS case '{case_id}' matched bool output at {output_path}.",
            diagnostics={"output_path": output_path},
        )
    return _case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path="bool-output",
        message=f"FAIL case '{case_id}' bool outputs differ at {output_path}.",
        diagnostics={
            "failure_stage": "bool_equal",
            "output_path": output_path,
        },
    )


def _compare_integer_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
    output_dtype: str,
    bound: int,
    comparison_path: str,
) -> CaseCompareResult:
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    diff = (actual_cast.to(dtype=torch.int64) - golden_cpu.to(dtype=torch.int64)).abs()
    max_diff = int(diff.max().item()) if diff.numel() > 0 else 0
    if max_diff <= bound:
        return _case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"PASS case '{case_id}' matched integer output at {output_path}.",
            diagnostics={
                "output_path": output_path,
                "max_abs_diff": max_diff,
                "error_bound": bound,
            },
        )
    flat_index = int(torch.argmax(diff).item())
    return _case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=(
            f"FAIL case '{case_id}' integer comparison exceeded the error bound at {output_path}: "
            f"max_abs_diff={max_diff}, bound={bound}."
        ),
        diagnostics={
            "failure_stage": "integer_error_bound",
            "output_path": output_path,
            "max_abs_diff": max_diff,
            "max_abs_diff_index": _unravel_index(flat_index, tuple(golden_cpu.shape)),
            "error_bound": bound,
        },
    )


def _compare_floating_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
    output_dtype: str,
) -> CaseCompareResult:
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    finite_mask = torch.isfinite(actual_cast) & torch.isfinite(golden_cpu)
    finite_count = int(torch.count_nonzero(finite_mask).item())
    thresholds = _thresholds_for_output_dtype(output_dtype)
    max_abs_diff = 0.0
    max_abs_diff_index: tuple[int, ...] | tuple[()] = ()
    if finite_count > 0:
        actual_finite, golden_finite = _finite_float_views(actual_cast, golden_cpu, finite_mask)
        diff = (actual_finite - golden_finite).abs()
        abs_golden = golden_finite.abs()
        error_cap = thresholds["atol"] + thresholds["rtol"] * abs_golden
        cap_mask = diff <= error_cap
        max_abs_diff = float(diff.max().item())
        max_flat_index = int(torch.argmax(diff).item())
        finite_indices = finite_mask.nonzero(as_tuple=False)
        max_abs_diff_index = _indices_tensor_to_tuple(finite_indices[max_flat_index])
        if not bool(torch.all(cap_mask).item()):
            failing_flat_index = int(torch.nonzero(~cap_mask, as_tuple=False)[0].item())
            failing_index = _indices_tensor_to_tuple(finite_indices[failing_flat_index])
            return _case_result(
                passed=False,
                case_id=case_id,
                compute=compute,
                input_info=input_info,
                output_dtype=output_dtype,
                comparison_path="floating-point-compute",
                message=(
                    f"FAIL case '{case_id}' max error cap failed at {output_path}: "
                    f"diff={float(diff[failing_flat_index].item())}, "
                    f"cap={float(error_cap[failing_flat_index].item())}."
                ),
                diagnostics={
                    "failure_stage": "max_error_cap",
                    "output_path": output_path,
                    "finite_count": finite_count,
                    "max_abs_diff": max_abs_diff,
                    "max_abs_diff_index": max_abs_diff_index,
                    "first_failing_index": failing_index,
                    "max_error_cap_at_index": float(error_cap[failing_flat_index].item()),
                    "output_dtype": output_dtype,
                    "thresholds": dict(thresholds),
                },
            )
        small_mask = abs_golden < thresholds["small_value_threshold"]
        matched_small = diff <= thresholds["small_value_error"]
        matched_normal = diff / (abs_golden + 1e-7) <= thresholds["rel_threshold"]
        matched = torch.where(small_mask, matched_small, matched_normal)
        matched_ratio = float(matched.to(dtype=torch.float32).mean().item())
        mere = float((diff / (abs_golden + 1e-7)).mean().item())
        if matched_ratio < 0.9:
            return _case_result(
                passed=False,
                case_id=case_id,
                compute=compute,
                input_info=input_info,
                output_dtype=output_dtype,
                comparison_path="floating-point-compute",
                message=(
                    f"FAIL case '{case_id}' matched ratio failed at {output_path}: "
                    f"matched_ratio={matched_ratio}."
                ),
                diagnostics={
                    "failure_stage": "matched_ratio",
                    "output_path": output_path,
                    "finite_count": finite_count,
                    "matched_ratio": matched_ratio,
                    "mere": mere,
                    "mere_threshold": thresholds["rel_threshold"],
                    "max_abs_diff": max_abs_diff,
                    "max_abs_diff_index": max_abs_diff_index,
                    "output_dtype": output_dtype,
                    "thresholds": dict(thresholds),
                },
            )
        if mere >= thresholds["rel_threshold"]:
            return _case_result(
                passed=False,
                case_id=case_id,
                compute=compute,
                input_info=input_info,
                output_dtype=output_dtype,
                comparison_path="floating-point-compute",
                message=(
                    f"FAIL case '{case_id}' MERE failed at {output_path}: "
                    f"mere={mere}, threshold={thresholds['rel_threshold']}."
                ),
                diagnostics={
                    "failure_stage": "mere",
                    "output_path": output_path,
                    "finite_count": finite_count,
                    "matched_ratio": matched_ratio,
                    "mere": mere,
                    "mere_threshold": thresholds["rel_threshold"],
                    "max_abs_diff": max_abs_diff,
                    "max_abs_diff_index": max_abs_diff_index,
                    "output_dtype": output_dtype,
                    "thresholds": dict(thresholds),
                },
            )
        return _case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="floating-point-compute",
            message=f"PASS case '{case_id}' matched floating-point output at {output_path}.",
            diagnostics={
                "output_path": output_path,
                "finite_count": finite_count,
                "matched_ratio": matched_ratio,
                "mere": mere,
                "mere_threshold": thresholds["rel_threshold"],
                "max_abs_diff": max_abs_diff,
                "max_abs_diff_index": max_abs_diff_index,
                "output_dtype": output_dtype,
                "thresholds": dict(thresholds),
            },
        )
    return _case_result(
        passed=True,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path="floating-point-compute",
        message=f"PASS case '{case_id}' had no finite floating-point elements at {output_path}.",
        diagnostics={
            "output_path": output_path,
            "finite_count": 0,
            "matched_ratio": 1.0,
            "mere": 0.0,
            "mere_threshold": thresholds["rel_threshold"],
            "output_dtype": output_dtype,
            "thresholds": dict(thresholds),
        },
    )


def _finite_float_views(
    actual_cast: torch.Tensor,
    golden_cpu: torch.Tensor,
    finite_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_finite = actual_cast[finite_mask]
    golden_finite = golden_cpu[finite_mask]
    if actual_finite.is_complex() or golden_finite.is_complex():
        return (
            torch.view_as_real(actual_finite.to(dtype=torch.complex64))
            .reshape(-1)
            .to(dtype=torch.float32),
            torch.view_as_real(golden_finite.to(dtype=torch.complex64))
            .reshape(-1)
            .to(dtype=torch.float32),
        )
    return (
        actual_finite.to(dtype=torch.float32),
        golden_finite.to(dtype=torch.float32),
    )


def _thresholds_for_output_dtype(output_dtype: str) -> dict[str, float]:
    matched_key = _threshold_key(output_dtype)
    matched = _MATCH_THRESHOLDS.get(matched_key, _MATCH_THRESHOLDS["fallback"])
    max_error = _MAX_ERROR_THRESHOLDS.get(matched_key, _MAX_ERROR_THRESHOLDS["fallback"])
    return {
        "small_value_threshold": float(matched["small_value_threshold"]),
        "small_value_error": float(matched["small_value_error"]),
        "rel_threshold": float(matched["rel_threshold"]),
        "atol": float(max_error["atol"]),
        "rtol": float(max_error["rtol"]),
    }


def _threshold_key(output_dtype: str) -> str:
    if output_dtype.startswith("float8_e4m3"):
        return "float8_e4m3"
    if output_dtype.startswith("float8_e5m2"):
        return "float8_e5m2"
    if output_dtype in {"float16", "bfloat16", "float32", "hifloat32"}:
        return output_dtype
    return "fallback"


def _select_comparison_path(
    *,
    compute: bool,
    input_type: str,
    output_dtype: str,
) -> str | None:
    if not compute:
        return "non-compute"
    if output_dtype == "bool":
        return "bool-output"
    if output_dtype in _INTEGER_FAMILY:
        if input_type == "float":
            return "quantized-fp-to-int"
        if input_type in {"int", "no_tensor"}:
            return "integer-compute"
        return None
    if output_dtype in _OUTPUT_FLOAT_FAMILY:
        return "floating-point-compute"
    return None


def _infer_input_info(inputs: object) -> _InputInfo:
    direct_values = _direct_input_values(inputs)
    direct_tensors = [value for value in direct_values if isinstance(value, torch.Tensor)]
    if direct_tensors:
        dtype_name = min(
            (_dtype_name(tensor.dtype) for tensor in direct_tensors),
            key=lambda name: _DTYPE_PRIORITY.get(name, len(_DTYPE_PRIORITY)),
        )
        return _InputInfo(
            input_type="float" if dtype_name in _INPUT_FLOAT_FAMILY else "int",
            input_dtype=dtype_name,
        )
    for value in direct_values:
        if isinstance(value, (list, tuple)):
            sequence_values = cast(list[object] | tuple[object, ...], value)
            tensor_items = [item for item in sequence_values if isinstance(item, torch.Tensor)]
            if tensor_items and len(tensor_items) == len(sequence_values):
                first_tensor = tensor_items[0]
                dtype_name = _dtype_name(first_tensor.dtype)
                return _InputInfo(
                    input_type="float" if dtype_name in _INPUT_FLOAT_FAMILY else "int",
                    input_dtype=dtype_name,
                )
    return _InputInfo(input_type="no_tensor", input_dtype=None)


def _direct_input_values(inputs: object) -> list[object]:
    if isinstance(inputs, Mapping):
        return list(cast(Mapping[object, object], inputs).values())
    if isinstance(inputs, tuple):
        return list(cast(tuple[object, ...], inputs))
    if isinstance(inputs, list):
        return list(cast(list[object], inputs))
    if inputs is None:
        return []
    return [inputs]


def _coerce_output_leaf(value: object) -> torch.Tensor | None:
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


def _nan_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() or tensor.is_complex():
        return torch.isnan(tensor)
    return torch.zeros_like(tensor, dtype=torch.bool)


def _inf_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() or tensor.is_complex():
        return torch.isinf(tensor)
    return torch.zeros_like(tensor, dtype=torch.bool)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _is_sequence_output(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _indices_tensor_to_tuple(indices: torch.Tensor) -> tuple[int, ...]:
    flat_indices = indices.reshape(-1)
    return tuple(int(flat_indices[index].item()) for index in range(flat_indices.numel()))


def _unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...] | tuple[()]:
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


def _shape_mismatch_result(
    *,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
    output_path: str,
    output_dtype: str,
    comparison_path: str,
    actual_shape: tuple[int, ...],
    golden_shape: tuple[int, ...],
) -> CaseCompareResult:
    return _case_result(
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


def _case_result(
    *,
    passed: bool,
    case_id: str,
    compute: bool,
    input_info: _InputInfo,
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
        },
    )
