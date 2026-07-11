import os
import unittest
from unittest.mock import patch

from tests.run_skill_test_utils import load_npu_compare_module


class NpuCompareTests(unittest.TestCase):
    def test_compare_case_result_non_compute_avoids_python_storage_byte_materialization(self) -> None:
        module = load_npu_compare_module()
        import torch

        actual = torch.tensor([1.0, 2.0], dtype=torch.float32)
        golden = actual.clone()

        with patch.object(
            torch.storage.UntypedStorage,
            "__iter__",
            side_effect=AssertionError("non-compute compare should not iterate storage in Python"),
        ):
            result = module.compare_case_result(
                case_id="case-non-compute-no-python-bytes",
                actual=actual,
                golden=golden,
                inputs=(),
                compute=False,
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.comparison_path, "non-compute")

    def test_compare_case_result_non_compute_accepts_identical_nan_bit_patterns(self) -> None:
        module = load_npu_compare_module()
        import torch

        value = torch.tensor([float("nan")], dtype=torch.float32)
        result = module.compare_case_result(
            case_id="case-non-compute-nan",
            actual=value.clone(),
            golden=value.clone(),
            inputs=(),
            compute=False,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.comparison_path, "non-compute")

    def test_compare_case_result_non_compute_rejects_signed_zero_bit_difference(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-non-compute-signed-zero",
            actual=torch.tensor([0.0], dtype=torch.float32),
            golden=torch.tensor([-0.0], dtype=torch.float32),
            inputs=(),
            compute=False,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.comparison_path, "non-compute")
        self.assertEqual(result.diagnostics["failure_stage"], "binary_equal")

    def test_compare_case_result_routes_float_to_int_as_quantized_path(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-a",
            actual=torch.tensor([2, 5], dtype=torch.int32),
            golden=torch.tensor([1, 5], dtype=torch.int32),
            inputs=(torch.tensor([0.5, 1.5], dtype=torch.float16),),
            compute=True,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.comparison_path, "quantized-fp-to-int")
        self.assertEqual(result.input_type, "float")
        self.assertEqual(result.output_dtype, "int32")

    def test_compare_case_result_reports_detailed_float_failure(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-a",
            actual=torch.tensor([1.5], dtype=torch.float32),
            golden=torch.tensor([1.0], dtype=torch.float32),
            inputs=(torch.tensor([1.0], dtype=torch.float32),),
            compute=True,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.comparison_path, "floating-point-compute")
        self.assertEqual(result.diagnostics["failure_stage"], "max_error_cap")
        self.assertEqual(result.diagnostics["output_path"], "output")
        self.assertEqual(result.diagnostics["output_dtype"], "float32")
        self.assertEqual(result.diagnostics["accuracy_mode"], "npu-contract")
        self.assertIn("thresholds", result.diagnostics)
        self.assertIn("case-a", result.message)

    def test_compare_case_result_dtype_close_uses_env_tolerance_overrides(self) -> None:
        module = load_npu_compare_module()
        import torch

        with patch.dict(
            os.environ,
            {
                "HELIX_RUN_TEST_ACCURACY_MODE": "dtype-close",
                "HELIX_RUN_TEST_ATOL": "0",
                "HELIX_RUN_TEST_RTOL": "0.01",
            },
            clear=False,
        ):
            result = module.compare_case_result(
                case_id="case-dtype-close-env",
                actual=torch.tensor([1.005], dtype=torch.float32),
                golden=torch.tensor([1.0], dtype=torch.float32),
                inputs=(torch.tensor([1.0], dtype=torch.float32),),
                compute=True,
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.diagnostics["accuracy_mode"], "dtype-close")
        self.assertEqual(result.diagnostics["atol"], 0.0)
        self.assertEqual(result.diagnostics["rtol"], 0.01)

    def test_compare_case_result_dtype_close_reports_assert_close_failure(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-dtype-close-fail",
            actual=torch.tensor([1.1], dtype=torch.float32),
            golden=torch.tensor([1.0], dtype=torch.float32),
            inputs=(torch.tensor([1.0], dtype=torch.float32),),
            compute=True,
            accuracy_mode="dtype-close",
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.diagnostics["accuracy_mode"], "dtype-close")
        self.assertEqual(result.diagnostics["failure_stage"], "assert_close")
        self.assertEqual(result.diagnostics["tensor_shape"], (1,))
        self.assertEqual(result.diagnostics["atol"], 1e-5)
        self.assertEqual(result.diagnostics["rtol"], 1e-4)
        self.assertIn("assert_close_message", result.diagnostics)

    def test_compare_case_result_dtype_close_keeps_dtype_strict(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-dtype-close-dtype",
            actual=torch.tensor([1.0], dtype=torch.float64),
            golden=torch.tensor([1.0], dtype=torch.float32),
            inputs=(torch.tensor([1.0], dtype=torch.float32),),
            compute=True,
            accuracy_mode="dtype-close",
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.diagnostics["accuracy_mode"], "dtype-close")
        self.assertEqual(result.diagnostics["failure_stage"], "dtype_mismatch")

    def test_compare_case_result_handles_bool_output_with_strict_equality(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-a",
            actual=torch.tensor([True, False], dtype=torch.bool),
            golden=torch.tensor([True, True], dtype=torch.bool),
            inputs=(torch.tensor([1], dtype=torch.int32),),
            compute=True,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.comparison_path, "bool-output")
        self.assertEqual(result.diagnostics["failure_stage"], "bool_equal")

    def test_compare_case_result_uses_floating_point_path_for_no_tensor_inputs(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-a",
            actual=torch.tensor([1.0], dtype=torch.float32),
            golden=torch.tensor([1.0], dtype=torch.float32),
            inputs=(),
            compute=True,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.input_type, "no_tensor")
        self.assertEqual(result.comparison_path, "floating-point-compute")

    def test_compare_case_result_supports_complex_tensor_outputs(self) -> None:
        module = load_npu_compare_module()
        import torch

        result = module.compare_case_result(
            case_id="case-complex-tensor",
            actual=torch.tensor([1 + 2j], dtype=torch.complex64),
            golden=torch.tensor([1 + 2j], dtype=torch.complex64),
            inputs=(torch.tensor([1.0], dtype=torch.float32),),
            compute=True,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.comparison_path, "floating-point-compute")
        self.assertEqual(result.output_dtype, "complex64")

    def test_compare_case_result_supports_complex_scalar_outputs(self) -> None:
        module = load_npu_compare_module()

        result = module.compare_case_result(
            case_id="case-complex-scalar",
            actual=1 + 2j,
            golden=1 + 2j,
            inputs=(),
            compute=True,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.comparison_path, "floating-point-compute")
        self.assertIn(result.output_dtype, {"complex64", "complex128"})

    def test_compare_result_payloads_rejects_legacy_results_format(self) -> None:
        module = load_npu_compare_module()

        result = module.compare_result_payloads(
            {"results": [1]},
            {"results": [1]},
        )

        self.assertFalse(result.passed)
        self.assertIn("legacy payload format", result.message)
        self.assertEqual(result.diagnostics["accuracy_mode"], "npu-contract")

    def test_compare_result_payloads_accepts_dtype_close_mode(self) -> None:
        module = load_npu_compare_module()
        import torch

        oracle = {
            "compute": True,
            "cases": [
                {
                    "id": "case-a",
                    "inputs": (torch.tensor([1.0], dtype=torch.float32),),
                    "result": torch.tensor([1.0], dtype=torch.float32),
                }
            ],
        }
        actual = {
            "compute": True,
            "cases": [
                {
                    "id": "case-a",
                    "inputs": (torch.tensor([1.0], dtype=torch.float32),),
                    "result": torch.tensor([1.00005], dtype=torch.float32),
                }
            ],
        }

        result = module.compare_result_payloads(oracle, actual, accuracy_mode="dtype-close")

        self.assertTrue(result.passed)
        self.assertEqual(result.diagnostics["accuracy_mode"], "dtype-close")
        self.assertEqual(result.case_results[0].diagnostics["accuracy_mode"], "dtype-close")


if __name__ == "__main__":
    unittest.main()
