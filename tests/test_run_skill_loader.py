import ast
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skill_loader import (
    load_operator_eval_script_module,
    load_skill_script_module,
    operator_eval_script_path,
    skill_script_path,
)
import triton_agent.optimize.naming as optimize_naming
import triton_agent.optimize.pt_cleanup as optimize_pt_cleanup
from triton_agent.optimize.models import BaselineState, OptimizeCheckResult, RoundState


def _top_level_defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


class RunSkillLoaderTests(unittest.TestCase):
    def test_test_runner_wrapper_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("triton_agent.test_runner"))

    def test_bench_runner_wrapper_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("triton_agent.bench_runner"))

    def test_operator_eval_script_path_points_to_run_scripts(self) -> None:
        path = operator_eval_script_path("run-command")
        self.assertEqual(path.name, "run-command.py")
        self.assertEqual(path.parent.name, "scripts")
        self.assertEqual(path.parent.parent.name, "triton-npu-run-eval")

    def test_load_operator_eval_script_module_returns_cached_module(self) -> None:
        first = load_operator_eval_script_module("test_runner")
        second = load_operator_eval_script_module("test_runner")
        self.assertIs(first, second)
        self.assertTrue(hasattr(first, "run_local_test"))

    def test_skill_script_path_points_to_optimize_check_script(self) -> None:
        path = skill_script_path("triton-npu-optimize-check", "optimize_check")
        self.assertEqual(path.name, "optimize_check.py")
        self.assertEqual(path.parent.name, "scripts")
        self.assertEqual(path.parent.parent.name, "triton-npu-optimize-check")

    def test_load_skill_script_module_returns_cached_module(self) -> None:
        first = load_skill_script_module("triton-npu-optimize-check", "optimize_check")
        second = load_skill_script_module("triton-npu-optimize-check", "optimize_check")
        self.assertIs(first, second)
        self.assertTrue(hasattr(first, "check_baseline"))
        self.assertTrue(hasattr(first, "check_round"))

    def test_optimize_runtime_models_reuse_skill_contract_classes(self) -> None:
        module = load_skill_script_module("triton-npu-optimize-check", "optimize_check")

        self.assertIs(module.OptimizeCheckResult, OptimizeCheckResult)
        self.assertIs(module.BaselineState, BaselineState)
        self.assertIs(module.RoundState, RoundState)

    def test_optimize_runtime_naming_helpers_reuse_skill_contract_functions(self) -> None:
        module = load_skill_script_module("triton-npu-optimize-check", "optimize_check")

        self.assertIs(optimize_naming.expected_round_operator_name, module.expected_round_operator_name)
        self.assertIs(optimize_naming.expected_round_perf_name, module.expected_round_perf_name)
        self.assertIs(optimize_naming.resolve_round_operator_file, module.resolve_round_operator_file)
        self.assertIs(optimize_naming.resolve_round_perf_file, module.resolve_round_perf_file)

    def test_optimize_runtime_pt_cleanup_helpers_reuse_skill_contract_functions(self) -> None:
        module = load_skill_script_module("triton-npu-optimize-check", "optimize_check")

        self.assertIs(optimize_pt_cleanup.cleanup_dir_pt_files, module.cleanup_dir_pt_files)

    def test_run_skill_scripts_do_not_import_triton_agent(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts"
        for path in sorted(scripts_dir.glob("*.py")):
            with self.subTest(path=path.name):
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("import triton_agent", content)
                self.assertNotIn("from triton_agent", content)

    def test_optimize_check_script_does_not_import_runtime_sources_directly(self) -> None:
        path = skill_script_path("triton-npu-optimize-check", "optimize_check")
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("from src.", content)
        self.assertNotIn("import src.", content)

    def test_optimize_check_skill_scripts_do_not_import_triton_agent(self) -> None:
        scripts_dir = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-optimize-check"
            / "scripts"
        )
        for path in sorted(scripts_dir.glob("*.py")):
            with self.subTest(path=path.name):
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("import triton_agent", content)
                self.assertNotIn("from triton_agent", content)

    def test_run_runtime_only_exposes_skill_runtime_helpers(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        self.assertTrue(hasattr(module, "run_streaming_process"))
        self.assertTrue(hasattr(module, "run_buffered_process"))
        self.assertFalse(hasattr(module, "run_process"))
        self.assertFalse(hasattr(module, "run_interactive_process"))

    def test_run_command_and_runtime_use_shared_result_payload_helper(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts"
        self.assertTrue((scripts_dir / "result_payload.py").is_file())
        self.assertNotIn("ResultPayload", _top_level_defined_names(scripts_dir / "run-command.py"))
        self.assertNotIn("ResultPayload", _top_level_defined_names(scripts_dir / "run_runtime.py"))
        self.assertNotIn("make_result", _top_level_defined_names(scripts_dir / "run_runtime.py"))

    def test_bench_runner_no_longer_uses_globals_service_locator(self) -> None:
        path = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts" / "bench_runner.py"
        content = path.read_text(encoding="utf-8")

        self.assertNotIn("globals()[name]", content)
        self.assertNotIn("_FACADE_COMPAT_EXPORTS", content)

    def test_bench_runner_submodules_use_explicit_dependency_protocols(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts"
        msprof_content = (scripts_dir / "bench_runner_msprof.py").read_text(encoding="utf-8")
        standalone_content = (scripts_dir / "bench_runner_standalone.py").read_text(encoding="utf-8")

        self.assertNotIn("deps: Any", msprof_content)
        self.assertNotIn("deps: Any", standalone_content)


if __name__ == "__main__":
    unittest.main()
