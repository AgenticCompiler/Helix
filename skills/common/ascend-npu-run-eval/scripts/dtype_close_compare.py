from __future__ import annotations

import torch

from npu_compare_common import (
    CaseCompareResult,
    InputInfo,
    case_result,
    dtype_close_tolerances_for_output_dtype,
    max_abs_diff,
)


def compare_dtype_close_output(
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
    if comparison_path == "bool-output":
        if torch.equal(actual_cpu, golden_cpu):
            return case_result(
                passed=True,
                case_id=case_id,
                compute=compute,
                input_info=input_info,
                output_dtype=output_dtype,
                comparison_path=comparison_path,
                message=f"PASS case '{case_id}' matched bool output at {output_path}.",
                diagnostics={"output_path": output_path},
            )
        return case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"FAIL case '{case_id}' bool outputs differ at {output_path}.",
            diagnostics={"failure_stage": "bool_equal", "output_path": output_path},
        )
    if comparison_path in {"integer-compute", "quantized-fp-to-int"}:
        bound = 0
        diff = (actual_cpu.to(dtype=torch.int64) - golden_cpu.to(dtype=torch.int64)).abs()
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
                diagnostics={
                    "output_path": output_path,
                    "max_abs_diff": max_diff,
                    "error_bound": bound,
                },
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
                "max_abs_diff_index": (flat_index,) if len(golden_cpu.shape) <= 1 else None,
                "error_bound": bound,
            },
        )
    tolerances = dtype_close_tolerances_for_output_dtype(output_dtype)
    diagnostics = {
        "output_path": output_path,
        "output_dtype": output_dtype,
        "tensor_shape": tuple(golden_cpu.shape),
        "atol": tolerances["atol"],
        "rtol": tolerances["rtol"],
        "max_abs_diff": max_abs_diff(actual_cpu, golden_cpu),
    }
    try:
        torch.testing.assert_close(
            actual_cpu,
            golden_cpu,
            atol=tolerances["atol"],
            rtol=tolerances["rtol"],
            equal_nan=True,
        )
    except AssertionError as exc:
        return case_result(
            passed=False,
            case_id=case_id,
            compute=compute,
            input_info=input_info,
            output_dtype=output_dtype,
            comparison_path=comparison_path,
            message=f"FAIL case '{case_id}' assert_close failed at {output_path}.",
            diagnostics={
                **diagnostics,
                "failure_stage": "assert_close",
                "assert_close_message": str(exc),
            },
        )
    return case_result(
        passed=True,
        case_id=case_id,
        compute=compute,
        input_info=input_info,
        output_dtype=output_dtype,
        comparison_path=comparison_path,
        message=f"PASS case '{case_id}' matched dtype-close output at {output_path}.",
        diagnostics=diagnostics,
    )
