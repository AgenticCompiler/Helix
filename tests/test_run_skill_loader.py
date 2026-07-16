import ast
from dataclasses import fields
import importlib.util
import sys
from typing import get_type_hints
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.skills.loader import (
    load_operator_eval_script_module,
    load_skill_script_module,
    operator_eval_script_path,
    skill_script_path,
)
import helix.optimize.naming as optimize_naming
import helix.optimize.pt_cleanup as optimize_pt_cleanup
from helix.optimize.models import BaselineState, OptimizeCheckResult, RoundState


def _top_level_defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


class RunSkillLoaderTests(unittest.TestCase):
    def test_test_runner_wrapper_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("helix.test_runner"))

    def test_bench_runner_wrapper_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("helix.bench_runner"))

    def test_operator_eval_script_path_points_to_run_eval_cli(self) -> None:
        path = operator_eval_script_path("cli")
        self.assertEqual(path.name, "cli.py")
        self.assertEqual(path.parent.name, "scripts")
        self.assertEqual(path.parent.parent.name, "ascend-npu-run-eval")

    def test_load_operator_eval_script_module_returns_cached_module(self) -> None:
        first = load_operator_eval_script_module("run_test_api")
        second = load_operator_eval_script_module("run_test_api")
        self.assertIs(first, second)
        self.assertTrue(hasattr(first, "run_local_test"))

    def test_skill_script_path_points_to_optimize_state_cli_entrypoint(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "cli.py"
        )
        self.assertTrue(expected.exists())
        path = skill_script_path("ascend-npu-optimize-state", "cli")
        self.assertEqual(path.name, "cli.py")
        self.assertEqual(path.parent.name, "scripts")
        self.assertEqual(path.parent.parent.name, "ascend-npu-optimize-state")

    def test_skill_script_path_points_to_optimize_state_submit_round_script(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "state_manage"
            / "submit_round.py"
        )
        self.assertTrue(expected.exists())
        path = skill_script_path("ascend-npu-optimize-state", "state_manage/submit_round")
        self.assertEqual(path.name, "submit_round.py")
        self.assertEqual(path.parent.name, "state_manage")
        self.assertEqual(path.parent.parent.name, "scripts")

    def test_skill_script_path_points_to_optimize_state_submit_baseline_script(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "state_manage"
            / "submit_baseline.py"
        )
        self.assertTrue(expected.exists())
        path = skill_script_path("ascend-npu-optimize-state", "state_manage/submit_baseline")
        self.assertEqual(path.name, "submit_baseline.py")
        self.assertEqual(path.parent.name, "state_manage")
        self.assertEqual(path.parent.parent.name, "scripts")

    def test_skill_script_path_supports_nested_skill_relative_scripts(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "state_manage"
            / "state_machine.py"
        )
        path = skill_script_path("ascend-npu-optimize-state", "state_manage/state_machine")
        self.assertEqual(path, expected)

    def test_load_skill_script_module_returns_cached_split_modules(self) -> None:
        first = load_skill_script_module(
            "ascend-npu-optimize-state",
            "baseline/check",
        )
        second = load_skill_script_module(
            "ascend-npu-optimize-state",
            "baseline/check",
        )
        self.assertIs(first, second)
        self.assertTrue(hasattr(first, "load_baseline_state"))
        round_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "round/check",
        )
        self.assertTrue(hasattr(round_module, "check_round"))
        submit_round_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/submit_round",
        )
        self.assertTrue(hasattr(submit_round_module, "build_parser"))
        submit_baseline_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/submit_baseline",
        )
        self.assertTrue(hasattr(submit_baseline_module, "build_parser"))

    def test_load_skill_script_module_supports_nested_skill_relative_scripts(self) -> None:
        first = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/state_machine",
        )
        second = load_skill_script_module(
            "ascend-npu-optimize-state",
            "state_manage/state_machine",
        )
        self.assertIs(first, second)
        self.assertTrue(hasattr(first, "bootstrap_state"))

    def test_optimize_runtime_models_reuse_split_submit_skill_contract_classes(self) -> None:
        baseline_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "baseline/check",
        )
        round_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "round/check",
        )

        self.assertIs(baseline_module.BaselineState, BaselineState)
        self.assertIs(round_module.OptimizeCheckResult, OptimizeCheckResult)
        self.assertIs(round_module.RoundState, RoundState)

    def test_optimize_state_baseline_and_round_contracts_share_check_result_shape(self) -> None:
        baseline_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "baseline/check",
        )
        round_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "round/check",
        )

        baseline_result_type = baseline_module.OptimizeCheckResult
        round_result_type = round_module.OptimizeCheckResult

        self.assertEqual(
            [field.name for field in fields(baseline_result_type)],
            [field.name for field in fields(round_result_type)],
        )
        self.assertEqual(
            get_type_hints(baseline_result_type)["kind"],
            get_type_hints(round_result_type)["kind"],
        )

    def test_optimize_runtime_naming_helpers_reuse_optimize_state_round_contract_functions(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "round/check",
        )

        self.assertIs(optimize_naming.expected_round_operator_name, module.expected_round_operator_name)
        self.assertIs(optimize_naming.expected_round_perf_name, module.expected_round_perf_name)
        self.assertIs(optimize_naming.resolve_round_operator_file, module.resolve_round_operator_file)
        self.assertIs(optimize_naming.resolve_round_perf_file, module.resolve_round_perf_file)

    def test_optimize_runtime_pt_cleanup_helpers_reuse_optimize_state_round_contract_functions(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "round/check",
        )

        self.assertIs(optimize_pt_cleanup.cleanup_dir_pt_files, module.cleanup_dir_pt_files)

    def test_run_skill_scripts_do_not_import_helix(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "common" / "ascend-npu-run-eval" / "scripts"
        for path in sorted(scripts_dir.glob("*.py")):
            with self.subTest(path=path.name):
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("import helix", content)
                self.assertNotIn("from helix", content)

    def test_optimize_state_round_check_script_does_not_import_runtime_sources_directly(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "round"
            / "check.py"
        )
        self.assertTrue(expected.exists())
        path = skill_script_path("ascend-npu-optimize-state", "round/check")
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("from src.", content)
        self.assertNotIn("import src.", content)

    def test_optimize_state_submit_round_script_does_not_import_runtime_sources_directly(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "state_manage"
            / "submit_round.py"
        )
        self.assertTrue(expected.exists())
        path = skill_script_path("ascend-npu-optimize-state", "state_manage/submit_round")
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("from src.", content)
        self.assertNotIn("import src.", content)

    def test_optimize_state_submit_baseline_script_does_not_import_runtime_sources_directly(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "state_manage"
            / "submit_baseline.py"
        )
        self.assertTrue(expected.exists())
        path = skill_script_path("ascend-npu-optimize-state", "state_manage/submit_baseline")
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("from src.", content)
        self.assertNotIn("import src.", content)

    def test_optimize_state_skill_scripts_do_not_import_helix(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = repo_root / "skills" / "common" / "ascend-npu-optimize-state" / "scripts"
        self.assertTrue(scripts_dir.is_dir())
        for path in sorted(scripts_dir.rglob("*.py")):
            with self.subTest(path=path.relative_to(scripts_dir).as_posix()):
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("import helix", content)
                self.assertNotIn("from helix", content)

    def test_optimize_state_baseline_directory_keeps_only_baseline_specific_scripts(self) -> None:
        scripts_dir = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-optimize-state"
            / "scripts"
            / "baseline"
        )
        script_names = {path.name for path in scripts_dir.glob("*.py")}

        self.assertEqual(
            script_names,
            {
                "check.py",
                "contract.py",
            },
        )

    def test_run_runtime_only_exposes_skill_runtime_helpers(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        self.assertTrue(hasattr(module, "run_streaming_process"))
        self.assertTrue(hasattr(module, "run_buffered_process"))
        self.assertFalse(hasattr(module, "run_process"))
        self.assertFalse(hasattr(module, "run_interactive_process"))

    def test_run_command_and_runtime_use_shared_result_payload_helper(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "common" / "ascend-npu-run-eval" / "scripts"
        self.assertTrue((scripts_dir / "result_payload.py").is_file())
        self.assertNotIn("ResultPayload", _top_level_defined_names(scripts_dir / "cli.py"))
        self.assertNotIn("ResultPayload", _top_level_defined_names(scripts_dir / "run_runtime.py"))
        self.assertNotIn("make_result", _top_level_defined_names(scripts_dir / "run_runtime.py"))

    def test_bench_runner_no_longer_uses_globals_service_locator(self) -> None:
        path = Path(__file__).resolve().parents[1] / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "bench_runner.py"
        content = path.read_text(encoding="utf-8")

        self.assertNotIn("globals()[name]", content)
        self.assertNotIn("_FACADE_COMPAT_EXPORTS", content)

    def test_bench_runner_is_single_file_without_submodule_dependency_adapter(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "common" / "ascend-npu-run-eval" / "scripts"
        bench_runner = scripts_dir / "bench_runner.py"
        content = bench_runner.read_text(encoding="utf-8")

        self.assertFalse((scripts_dir / "bench_runner_deps.py").exists())
        self.assertFalse((scripts_dir / "bench_runner_msprof.py").exists())
        self.assertFalse((scripts_dir / "bench_runner_standalone.py").exists())
        self.assertNotIn("BenchRunnerDeps", content)


if __name__ == "__main__":
    unittest.main()
