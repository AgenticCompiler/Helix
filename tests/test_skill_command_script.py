import os
import subprocess
import shutil
import sys
import tempfile
import unittest
import importlib.util
import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

class SkillCommandScriptTests(unittest.TestCase):
    @unittest.skipIf(shutil.which("bash") is None, "requires bash")
    def test_skill_script_pyright_wrapper_requires_exactly_one_target(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run-skill-script-pyright.sh"
        )
        completed = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("<skill-script.py>", completed.stderr)
        self.assertNotIn("[<skill-script.py> ...]", completed.stderr)

    @unittest.skipIf(shutil.which("bash") is None, "requires bash")
    def test_skill_script_pyright_wrapper_rejects_multiple_targets(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run-skill-script-pyright.sh"
        )
        completed = subprocess.run(
            [
                "bash",
                str(script),
                "skills/triton-npu-run-eval/scripts/bench_runner.py",
                "skills/triton-npu-run-eval/scripts/profile_runner.py",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stderr)

    def test_render_result_accepts_skill_result_payload(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "run-command.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout = StringIO()
        stderr = StringIO()
        try:
            sys.stdout = stdout
            sys.stderr = stderr
            module._render_result(
                {
                    "return_code": 0,
                    "stdout": "skill stdout\n",
                    "stderr": "skill stderr\n",
                    "stalled": False,
                    "session_id": None,
                },
                show_output=False,
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        self.assertEqual(stdout.getvalue(), "skill stdout\n")
        self.assertEqual(stderr.getvalue(), "skill stderr\n")

    def test_script_run_test_prints_hint_for_differential_result(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "run-command.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "kernel.py"
            test_file = root / "differential_test_kernel.py"
            archive = root / "kernel_result.pt"
            operator.write_text("print('x')\n", encoding="utf-8")
            test_file.write_text("# test-mode: differential\nprint('test')\n", encoding="utf-8")

            def fake_run_local_test(
                test_path: Path,
                operator_path: Path,
                test_mode: str,
            ) -> tuple[dict[str, object], Path]:
                self.assertEqual(test_path, test_file.resolve())
                self.assertEqual(operator_path, operator.resolve())
                self.assertEqual(test_mode, "differential")
                return (
                    {
                        "return_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "stalled": False,
                        "session_id": None,
                    },
                    archive,
                )

            stdout = StringIO()
            stderr = StringIO()
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                with patch.object(
                    module,
                    "_load_test_functions",
                    return_value=(
                        lambda _path: {"test-mode": "differential"},
                        fake_run_local_test,
                        lambda *_args, **_kwargs: None,
                    ),
                ):
                    exit_code = module.main(
                        [
                            "run-test",
                            "--test-file",
                            str(test_file),
                            "--operator-file",
                            str(operator),
                            "--test-mode",
                            "differential",
                        ]
                    )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            (
                "Return code: 0\n"
                f"Archived result: {archive}\n"
                "Hint: use `compare-result` to inspect this archived result instead of reading it directly.\n"
            ),
        )
        self.assertEqual(stderr.getvalue(), "")

    def test_script_runs_cli_help_without_installed_entrypoint(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "run-command.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("run-command.py", completed.stdout)
        self.assertNotIn("usage: triton-agent", completed.stdout)
        self.assertIn("run-test", completed.stdout)
        self.assertIn("compare-perf", completed.stdout)
        self.assertIn("profile-bench", completed.stdout)
        self.assertNotIn("optimize", completed.stdout)
        self.assertNotIn("gen-test", completed.stdout)

    def test_script_resolves_real_repo_root_when_called_through_symlink(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_skills = repo_root / "skills"
        source_script = source_skills / "triton-npu-run-eval" / "scripts" / "run-command.py"

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            symlinked_skills = workspace / "skills"
            try:
                symlinked_skills.symlink_to(source_skills, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")
            symlinked_script = symlinked_skills / "triton-npu-run-eval" / "scripts" / "run-command.py"

            completed = subprocess.run(
                [sys.executable, str(symlinked_script), "--help"],
                capture_output=True,
                text=True,
                cwd=workspace,
                check=False,
            )

        self.assertTrue(source_script.exists())
        self.assertEqual(completed.returncode, 0)
        self.assertIn("compare-result", completed.stdout)

    def test_script_exposes_standalone_run_test_help(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "run-command.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "run-test", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("usage: run-command.py run-test", completed.stdout)
        self.assertIn("--test-file", completed.stdout)
        self.assertIn("--operator-file", completed.stdout)
        self.assertIn("--keep-remote-workdir", completed.stdout)

    def test_script_exposes_profile_bench_help(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "run-command.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "profile-bench", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("--bench-file", completed.stdout)
        self.assertIn("--operator-file", completed.stdout)
        self.assertIn("--case-id", completed.stdout)
        self.assertIn("--bench", completed.stdout)
        self.assertNotIn("--kernel-name", completed.stdout)
        self.assertIn("--target-op", completed.stdout)
        self.assertIn("--keep-remote-workdir", completed.stdout)

    def test_load_compare_perf_function_reuses_perf_artifacts_implementation(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "run-command.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_compare_perf_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        compare_perf = module._load_compare_perf_function()

        self.assertEqual(compare_perf.__name__, "compare_perf_files")
        self.assertEqual(compare_perf.__module__, "perf_artifacts")

    def test_load_compare_result_functions_reuse_compare_payload_implementation(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "run-command.py"
        )
        spec = importlib.util.spec_from_file_location("run_command_compare_result_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        compare_result, compare_remote_result = module._load_compare_result_functions()

        self.assertEqual(compare_result.__name__, "compare_result_files")
        self.assertEqual(compare_result.__module__, "compare_result")
        self.assertEqual(compare_remote_result.__name__, "compare_remote_result_files")
        self.assertEqual(compare_remote_result.__module__, "compare_result")

    def test_optimize_check_script_help_runs_without_installed_entrypoint(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-optimize-check"
            / "scripts"
            / "optimize_check.py"
        )
        env = os.environ.copy()
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        env["PYTHONPATH"] = src_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("optimize_check.py", completed.stdout)
        self.assertIn("check-baseline", completed.stdout)
        self.assertIn("check-round", completed.stdout)

    def test_optimize_check_script_supports_runtime_without_pt_cleanup_module(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = (
            repo_root
            / "skills"
            / "triton-npu-optimize-check"
            / "scripts"
            / "optimize_check.py"
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            workspace = tmpdir / "workspace"
            runtime_root = tmpdir / "runtime"
            optimize_dir = runtime_root / "triton_agent" / "optimize"
            workspace.mkdir()
            (workspace / "baseline").mkdir()
            optimize_dir.mkdir(parents=True)

            (workspace / "baseline" / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "standalone",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (workspace / "baseline" / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (workspace / "baseline" / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (workspace / "baseline" / "test_result.pt").write_text("stub\n", encoding="utf-8")

            (runtime_root / "triton_agent" / "__init__.py").write_text("", encoding="utf-8")
            (optimize_dir / "__init__.py").write_text("", encoding="utf-8")
            (optimize_dir / "naming.py").write_text(
                "from pathlib import Path\n\n"
                "def expected_round_operator_name(workspace: Path) -> str:\n"
                "    return 'opt_kernel.py'\n\n"
                "def expected_round_perf_name(workspace: Path) -> str:\n"
                "    return 'opt_kernel_perf.txt'\n\n"
                "def resolve_round_operator_file(round_dir: Path):\n"
                "    return None\n\n"
                "def resolve_round_perf_file(round_dir: Path):\n"
                "    return None\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            script_dir = str(script.parent)
            runtime_path = str(runtime_root)
            pythonpath_entries = [runtime_path, script_dir]
            if env.get("PYTHONPATH"):
                pythonpath_entries.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(pythonpath_entries)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "check-baseline",
                    "--baseline-dir",
                    str(workspace / "baseline"),
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=workspace,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"decision": "pass"', completed.stdout)
            self.assertTrue((workspace / "baseline" / "test_result.pt").exists())


if __name__ == "__main__":
    unittest.main()
