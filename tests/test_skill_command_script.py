import subprocess
import sys
import tempfile
import unittest
import importlib.util
from io import StringIO
from pathlib import Path


class SkillCommandScriptTests(unittest.TestCase):
    def test_render_result_accepts_skill_result_payload(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "run-validation"
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

    def test_script_runs_cli_help_without_installed_entrypoint(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "skills"
            / "run-validation"
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
        self.assertIn("triton-agent", completed.stdout)
        self.assertIn("run-test", completed.stdout)
        self.assertIn("compare-perf", completed.stdout)
        self.assertNotIn("optimize", completed.stdout)
        self.assertNotIn("gen-test", completed.stdout)

    def test_script_resolves_real_repo_root_when_called_through_symlink(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_skills = repo_root / "skills"
        source_script = source_skills / "run-validation" / "scripts" / "run-command.py"

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            symlinked_skills = workspace / "skills"
            symlinked_skills.symlink_to(source_skills, target_is_directory=True)
            symlinked_script = symlinked_skills / "run-validation" / "scripts" / "run-command.py"

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
            / "run-validation"
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
        self.assertIn("--test-file", completed.stdout)
        self.assertIn("--operator-file", completed.stdout)
        self.assertIn("--keep-remote-workdir", completed.stdout)


if __name__ == "__main__":
    unittest.main()
