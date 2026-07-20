import ast
from dataclasses import fields
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.skills.loader import (
    load_operator_eval_script_module,
    load_skill_script_module,
    operator_eval_script_path,
    skill_script_path,
)
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

    def test_bench_execution_wrapper_module_has_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("helix.run_bench_execution"))

    def test_optimize_pure_forwarding_modules_have_been_removed(self) -> None:
        self.assertIsNone(importlib.util.find_spec("helix.optimize.checks"))
        self.assertIsNone(importlib.util.find_spec("helix.optimize.round_contract"))

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

    def test_optimize_runtime_models_are_owned_by_helix(self) -> None:
        baseline_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "baseline/check",
        )
        round_module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "round/check",
        )

        self.assertIsNot(baseline_module.BaselineState, BaselineState)
        self.assertIsNot(round_module.OptimizeCheckResult, OptimizeCheckResult)
        self.assertIsNot(round_module.RoundState, RoundState)
        self.assertEqual(
            [field.name for field in fields(baseline_module.BaselineState)],
            [field.name for field in fields(BaselineState)],
        )
        self.assertEqual(
            [field.name for field in fields(round_module.RoundState)],
            [field.name for field in fields(RoundState)],
        )

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

    def test_optimize_and_run_eval_facades_expose_explicit_api(self) -> None:
        optimize_api = load_skill_script_module("ascend-npu-optimize-state", "optimize_state_api")
        self.assertEqual(
            set(optimize_api.__all__),
            {
                "baseline_gate_issues",
                "best_completed_round_geomean_speedup",
                "bootstrap_state",
                "check_baseline",
                "check_round",
                "cleanup_dir_pt_files",
                "cleanup_pt_file",
                "cleanup_workspace_profile_artifacts",
                "count_completed_round_directories",
                "count_terminal_round_directories",
                "inspect_baseline_artifacts",
                "inspect_round_artifacts",
                "iter_terminal_round_directories",
                "load_baseline_state",
                "load_round_state",
                "load_state",
                "mark_baseline_passed",
                "ordinary_optimize_pt_cleanup_mode",
                "render_phase_summary",
                "resolve_round_operator_file",
                "resolve_round_perf_file",
            },
        )
        for script_name, expected_export in (
            ("compare_result_api", "compare_result_files"),
            ("perf_artifacts_api", "compare_perf_files"),
            ("run_simulator_api", "run_local_simulator"),
            ("remote_execution_env_api", "resolve_remote_execution"),
            ("run_runtime_api", "parse_remote_spec"),
        ):
            module = load_operator_eval_script_module(script_name)
            self.assertIn(expected_export, module.__all__)

    def test_helix_business_modules_load_skills_only_through_bridges(self) -> None:
        root = Path(__file__).resolve().parents[1] / "src" / "helix"
        allowed = {
            root / "skills" / "loader.py",
            *(root / "skill_bridges").glob("*.py"),
        }
        for path in sorted(root.rglob("*.py")):
            if path in allowed:
                continue
            content = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(root).as_posix()):
                self.assertNotIn("load_skill_script_module(", content)
                self.assertNotIn("load_operator_eval_script_module(", content)
                self.assertNotIn("operator_eval_script_path(", content)

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

    def test_bench_execution_no_longer_uses_globals_service_locator(self) -> None:
        path = Path(__file__).resolve().parents[1] / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "run_bench_execution.py"
        content = path.read_text(encoding="utf-8")

        self.assertNotIn("globals()[name]", content)
        self.assertNotIn("_FACADE_COMPAT_EXPORTS", content)

    def test_bench_execution_is_single_file_without_submodule_dependency_adapter(self) -> None:
        scripts_dir = Path(__file__).resolve().parents[1] / "skills" / "common" / "ascend-npu-run-eval" / "scripts"
        run_bench_execution = scripts_dir / "run_bench_execution.py"
        content = run_bench_execution.read_text(encoding="utf-8")

        self.assertFalse((scripts_dir / "bench_execution_deps.py").exists())
        self.assertFalse((scripts_dir / "bench_execution_msprof.py").exists())
        self.assertFalse((scripts_dir / "bench_execution_standalone.py").exists())
        self.assertNotIn("BenchRunnerDeps", content)


if __name__ == "__main__":
    unittest.main()
