from __future__ import annotations

import torch

from npu_compare_common import (
    CaseCompareResult,
    InputInfo,
    case_result,
    dtype_name,
    indices_tensor_to_tuple,
    inf_mask,
    nan_mask,
    shape_mismatch_result,
    thresholds_for_output_dtype,
    unravel_index,
)


def compare_npu_contract_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: InputInfo,
    output_path: str,
    output_dtype: str,
    comparison_path: str,
) -> CaseCompareResult:
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
    precheck_failure = _do_prechecks(
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
    input_info: InputInfo,
    output_path: str,
    output_dtype: str,
) -> CaseCompareResult:
    if tuple(actual_cpu.shape) != tuple(golden_cpu.shape):
        return shape_mismatch_result(
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
        return case_result(
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
                "expected_dtype": dtype_name(golden_cpu.dtype),
                "actual_dtype": dtype_name(actual_cpu.dtype),
            },
        )
    if torch.equal(actual_cpu.contiguous().view(torch.uint8), golden_cpu.contiguous().view(torch.uint8)):
        return case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="non-compute",
            message=f"PASS case '{case_id}' matched raw bits at {output_path}.",
            diagnostics={"output_path": output_path},
        )
    return case_result(
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
            "expected_dtype": dtype_name(golden_cpu.dtype),
            "actual_dtype": dtype_name(actual_cpu.dtype),
        },
    )


def _do_prechecks(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: InputInfo,
    output_path: str,
    output_dtype: str,
    comparison_path: str,
) -> CaseCompareResult | None:
    if tuple(actual_cpu.shape) != tuple(golden_cpu.shape):
        return shape_mismatch_result(
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
    nan_mask_actual = nan_mask(actual_cast)
    nan_mask_golden = nan_mask(golden_cpu)
    if not torch.equal(nan_mask_actual, nan_mask_golden):
        return case_result(
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
                "nan_mismatch_count": int(torch.count_nonzero(nan_mask_actual != nan_mask_golden).item()),
            },
        )
    inf_mask_actual = inf_mask(actual_cast)
    inf_mask_golden = inf_mask(golden_cpu)
    if not torch.equal(inf_mask_actual, inf_mask_golden):
        return case_result(
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
                "inf_mismatch_count": int(torch.count_nonzero(inf_mask_actual != inf_mask_golden).item()),
            },
        )
    if int(torch.count_nonzero(inf_mask_golden).item()) > 0 and not torch.equal(
        actual_cast[inf_mask_golden], golden_cpu[inf_mask_golden]
    ):
        return case_result(
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
                    torch.count_nonzero(actual_cast[inf_mask_golden] != golden_cpu[inf_mask_golden]).item()
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
    input_info: InputInfo,
    output_path: str,
    output_dtype: str,
) -> CaseCompareResult:
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    if torch.equal(actual_cast, golden_cpu):
        return case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path="bool-output",
            message=f"PASS case '{case_id}' matched bool output at {output_path}.",
            diagnostics={"output_path": output_path},
        )
    return case_result(
        passed=False,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path="bool-output",
        message=f"FAIL case '{case_id}' bool outputs differ at {output_path}.",
        diagnostics={"failure_stage": "bool_equal", "output_path": output_path},
    )


def _compare_integer_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: InputInfo,
    output_path: str,
    output_dtype: str,
    bound: int,
    comparison_path: str,
) -> CaseCompareResult:
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    diff = (actual_cast.to(dtype=torch.int64) - golden_cpu.to(dtype=torch.int64)).abs()
    max_diff = int(diff.max().item()) if diff.numel() > 0 else 0
    if max_diff <= bound:
        return case_result(
            passed=True,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"PASS case '{case_id}' matched integer output at {output_path}.",
            diagnostics={"output_path": output_path, "max_abs_diff": max_diff, "error_bound": bound},
        )
    flat_index = int(torch.argmax(diff).item())
    return case_result(
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
            "max_abs_diff_index": unravel_index(flat_index, tuple(golden_cpu.shape)),
            "error_bound": bound,
        },
    )


def _compare_floating_output(
    *,
    actual_cpu: torch.Tensor,
    golden_cpu: torch.Tensor,
    case_id: str,
    compute: bool,
    input_info: InputInfo,
    output_path: str,
    output_dtype: str,
) -> CaseCompareResult:
    actual_cast = actual_cpu.to(dtype=golden_cpu.dtype) if actual_cpu.dtype != golden_cpu.dtype else actual_cpu
    finite_mask = torch.isfinite(actual_cast) & torch.isfinite(golden_cpu)
    finite_count = int(torch.count_nonzero(finite_mask).item())
    thresholds = thresholds_for_output_dtype(output_dtype)
    max_diff = 0.0
    max_diff_index: tuple[int, ...] | tuple[()] = ()
    if finite_count > 0:
        actual_finite, golden_finite = _finite_float_views(actual_cast, golden_cpu, finite_mask)
        diff = (actual_finite - golden_finite).abs()
        abs_golden = golden_finite.abs()
        error_cap = thresholds["atol"] + thresholds["rtol"] * abs_golden
        cap_mask = diff <= error_cap
        max_diff = float(diff.max().item())
        max_flat_index = int(torch.argmax(diff).item())
        finite_indices = finite_mask.nonzero(as_tuple=False)
        max_diff_index = indices_tensor_to_tuple(finite_indices[max_flat_index])
        if not bool(torch.all(cap_mask).item()):
            failing_flat_index = int(torch.nonzero(~cap_mask, as_tuple=False)[0].item())
            failing_index = indices_tensor_to_tuple(finite_indices[failing_flat_index])
            return case_result(
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
                    "max_abs_diff": max_diff,
                    "max_abs_diff_index": max_diff_index,
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
            return case_result(
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
                    "max_abs_diff": max_diff,
                    "max_abs_diff_index": max_diff_index,
                    "output_dtype": output_dtype,
                    "thresholds": dict(thresholds),
                },
            )
        if mere >= thresholds["rel_threshold"]:
            return case_result(
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
                    "max_abs_diff": max_diff,
                    "max_abs_diff_index": max_diff_index,
                    "output_dtype": output_dtype,
                    "thresholds": dict(thresholds),
                },
            )
        return case_result(
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
                "max_abs_diff": max_diff,
                "max_abs_diff_index": max_diff_index,
                "output_dtype": output_dtype,
                "thresholds": dict(thresholds),
            },
        )
    return case_result(
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
            torch.view_as_real(actual_finite.to(dtype=torch.complex64)).reshape(-1).to(dtype=torch.float32),
            torch.view_as_real(golden_finite.to(dtype=torch.complex64)).reshape(-1).to(dtype=torch.float32),
        )
    return (actual_finite.to(dtype=torch.float32), golden_finite.to(dtype=torch.float32))
