import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SkillCommandScriptTests(unittest.TestCase):
    def test_script_runs_cli_help_without_installed_entrypoint(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "skills" / "scripts" / "run-command.py"
        )
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertIn("triton-agent", completed.stdout)
        self.assertIn("optimize", completed.stdout)

    def test_script_resolves_real_repo_root_when_called_through_symlink(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source_skills = repo_root / "skills"
        source_script = source_skills / "scripts" / "run-command.py"

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            symlinked_skills = workspace / "skills"
            symlinked_skills.symlink_to(source_skills, target_is_directory=True)
            symlinked_script = symlinked_skills / "scripts" / "run-command.py"

            completed = subprocess.run(
                [sys.executable, str(symlinked_script), "--help"],
                capture_output=True,
                text=True,
                cwd=workspace,
                check=False,
            )

        self.assertTrue(source_script.exists())
        self.assertEqual(completed.returncode, 0)
        self.assertIn("gen-test", completed.stdout)


if __name__ == "__main__":
    unittest.main()
