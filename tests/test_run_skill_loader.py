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

    def test_run_skill_scripts_do_not_import_triton_agent(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "triton-npu-run-eval" / "scripts"
        for path in sorted(scripts_dir.glob("*.py")):
            with self.subTest(path=path.name):
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("import triton_agent", content)
                self.assertNotIn("from triton_agent", content)

    def test_optimize_check_script_does_not_import_triton_agent(self) -> None:
        path = skill_script_path("triton-npu-optimize-check", "optimize_check")
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("import triton_agent", content)
        self.assertNotIn("from triton_agent", content)

    def test_run_runtime_only_exposes_skill_runtime_helpers(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        self.assertTrue(hasattr(module, "run_streaming_process"))
        self.assertTrue(hasattr(module, "run_buffered_process"))
        self.assertFalse(hasattr(module, "run_process"))
        self.assertFalse(hasattr(module, "run_interactive_process"))


if __name__ == "__main__":
    unittest.main()
