from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from dtype_close_compare import compare_dtype_close_output
from npu_compare_common import (
    ArtifactCompareResult,
    CaseCompareResult,
    DEFAULT_ACCURACY_MODE,
    InputInfo,
    case_result,
    coerce_output_leaf,
    current_accuracy_mode,
    dtype_name,
    infer_input_info,
    is_sequence_output,
    reset_current_accuracy_mode,
    resolve_accuracy_mode,
    select_comparison_path,
    set_current_accuracy_mode,
)
from npu_contract_compare import compare_npu_contract_output


@dataclass(frozen=True)
class _CaseRecord:
    case_id: str
    inputs: object
    result: object


def compare_case_result(
    *,
    case_id: str,
    actual: object,
    golden: object,
    inputs: object,
    compute: bool = True,
    accuracy_mode: str | None = None,
) -> CaseCompareResult:
    resolved_accuracy_mode = resolve_accuracy_mode(accuracy_mode)
    token = set_current_accuracy_mode(resolved_accuracy_mode)
    try:
        input_info = infer_input_info(inputs)
        leaf_results = _compare_value(
            actual=actual,
            golden=golden,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path="output",
        )
        if not leaf_results:
            return case_result(
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
        diagnostics: Mapping[str, object]
        if len(leaf_results) == 1:
            diagnostics = {**dict(first.diagnostics), "leaf_count": 1}
        else:
            diagnostics = {
                "case_id": case_id,
                "compute": compute,
                "input_type": input_info.input_type,
                "input_dtype": input_info.input_dtype,
                "output_path": "output",
                "leaf_count": len(leaf_results),
            }
        return case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"PASS case '{case_id}' matched across {len(leaf_results)} output leaf/leaves.",
            diagnostics=diagnostics,
        )
    finally:
        reset_current_accuracy_mode(token)


def compare_result_payloads(
    oracle_payload: object,
    actual_payload: object,
    accuracy_mode: str | None = None,
) -> ArtifactCompareResult:
    resolved_accuracy_mode = resolve_accuracy_mode(accuracy_mode)
    token = set_current_accuracy_mode(resolved_accuracy_mode)
    try:
        return _compare_result_payloads_in_current_mode(oracle_payload, actual_payload)
    finally:
        reset_current_accuracy_mode(token)


def _compare_result_payloads_in_current_mode(
    oracle_payload: object,
    actual_payload: object,
) -> ArtifactCompareResult:
    accuracy_mode = current_accuracy_mode()
    oracle_cases, oracle_error = _extract_case_records(oracle_payload, "oracle")
    if oracle_error is not None:
        return ArtifactCompareResult(
            passed=False,
            failed_case_count=1,
            case_results=(),
            message=f"FAIL: {oracle_error}",
            diagnostics={"payload": "oracle", "accuracy_mode": accuracy_mode},
        )
    actual_cases, actual_error = _extract_case_records(actual_payload, "actual_payload")
    if actual_error is not None:
        return ArtifactCompareResult(
            passed=False,
            failed_case_count=1,
            case_results=(),
            message=f"FAIL: {actual_error}",
            diagnostics={"payload": "actual_payload", "accuracy_mode": accuracy_mode},
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
            diagnostics={
                "oracle_case_count": len(oracle_cases),
                "actual_case_count": len(actual_cases),
                "accuracy_mode": accuracy_mode,
            },
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
            diagnostics={
                "oracle_compute": oracle_compute,
                "actual_compute": actual_compute,
                "accuracy_mode": accuracy_mode,
            },
        )
    case_results: list[CaseCompareResult] = []
    for index, (oracle_case, actual_case) in enumerate(zip(oracle_cases, actual_cases)):
        if oracle_case.case_id != actual_case.case_id:
            case_results.append(
                case_result(
                    passed=False,
                    case_id=oracle_case.case_id,
                    compute=oracle_compute,
                    input_info=infer_input_info(oracle_case.inputs),
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
                accuracy_mode=accuracy_mode,
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
            diagnostics={"case_count": len(case_results), "accuracy_mode": accuracy_mode},
        )
    message = (
        f"PASS: all {len(case_results)} case(s) matched the NPU accuracy contract."
        if accuracy_mode == DEFAULT_ACCURACY_MODE
        else f"PASS: all {len(case_results)} case(s) matched with accuracy mode '{accuracy_mode}'."
    )
    return ArtifactCompareResult(
        passed=True,
        failed_case_count=0,
        case_results=tuple(case_results),
        message=message,
        diagnostics={"case_count": len(case_results), "accuracy_mode": accuracy_mode},
    )


def format_artifact_compare_result(result: ArtifactCompareResult) -> str:
    lines = [result.message]
    if not result.passed:
        for case_result_item in result.case_results:
            if not case_result_item.passed:
                lines.append(case_result_item.message)
                diagnostics = case_result_item.diagnostics
                threshold_info = diagnostics.get("thresholds")
                if isinstance(threshold_info, Mapping):
                    lines.append(f"  thresholds={dict(cast(Mapping[object, object], threshold_info))}")
                lines.append(f"  diagnostics={dict(diagnostics)}")
    return "\n".join(lines)


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
        records.append(_CaseRecord(case_id=case_id, inputs=case_map["inputs"], result=case_map["result"]))
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
    input_info: InputInfo,
    output_path: str,
) -> list[CaseCompareResult]:
    if isinstance(golden, Mapping):
        if not isinstance(actual, Mapping):
            return [
                case_result(
                    passed=False,
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_dtype=None,
                    comparison_path="structure-mismatch",
                    message=f"FAIL case '{case_id}' expected mapping at {output_path}, got {type(actual).__name__}.",
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
                case_result(
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
            results.extend(
                _compare_value(
                    actual=actual_map[key],
                    golden=golden_map[key],
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_path=f"{output_path}.{key}",
                )
            )
        return results
    if is_sequence_output(golden):
        if not is_sequence_output(actual):
            return [
                case_result(
                    passed=False,
                    case_id=case_id,
                    compute=compute,
                    input_info=input_info,
                    output_dtype=None,
                    comparison_path="structure-mismatch",
                    message=f"FAIL case '{case_id}' expected sequence at {output_path}, got {type(actual).__name__}.",
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
                case_result(
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
    input_info: InputInfo,
    output_path: str,
) -> CaseCompareResult:
    actual_tensor = coerce_output_leaf(actual)
    golden_tensor = coerce_output_leaf(golden)
    if actual_tensor is None or golden_tensor is None:
        if not compute and actual == golden:
            return case_result(
                passed=True,
                case_id=case_id,
                compute=compute,
                input_info=input_info,
                output_dtype=None,
                comparison_path="non-compute",
                message=f"PASS case '{case_id}' matched non-tensor output at {output_path}.",
                diagnostics={"output_path": output_path},
            )
        return case_result(
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
    output_dtype = dtype_name(golden_cpu.dtype)
    comparison_path = select_comparison_path(
        compute=compute,
        input_type=input_info.input_type,
        output_dtype=output_dtype,
    )
    if comparison_path is None:
        return case_result(
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
    if current_accuracy_mode() == "dtype-close" and actual_cpu.dtype != golden_cpu.dtype:
        return case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"FAIL case '{case_id}' dtype mismatch at {output_path}.",
            diagnostics={
                "failure_stage": "dtype_mismatch",
                "output_path": output_path,
                "expected_dtype": dtype_name(golden_cpu.dtype),
                "actual_dtype": dtype_name(actual_cpu.dtype),
            },
        )
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    if current_accuracy_mode() == "dtype-close":
        return compare_dtype_close_output(
            actual_cpu=actual_cast,
            golden_cpu=golden_cpu,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_path=output_path,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
        )
    return compare_npu_contract_output(
        actual_cpu=actual_cpu,
        golden_cpu=golden_cpu,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_path=output_path,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
    )
