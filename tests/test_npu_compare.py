import unittest

from tests.run_skill_test_utils import load_npu_compare_module


class NpuCompareTests(unittest.TestCase):
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
        self.assertIn("thresholds", result.diagnostics)
        self.assertIn("case-a", result.message)

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


if __name__ == "__main__":
    unittest.main()
