import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_agent.models import CommandKind
from triton_agent.pattern_validation_loop.launcher import (
    build_optimize_batch_extra_flags,
    build_pattern_validation_loop_prompt,
    build_pattern_validation_loop_request,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class PatternValidationLoopLauncherTests(unittest.TestCase):
    def test_prompt_references_skill_helpers_and_loop_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp)
            synthesis = repo / "PERF_PATTERN_SYNTHESIS.md"
            synthesis.write_text("# Synthesis\n", encoding="utf-8")
            prompt = build_pattern_validation_loop_prompt(
                repo_path=repo,
                synthesis_path=synthesis,
                batch_dir=repo / "pattern-validation-batch",
                skills_workdir=repo / "pattern-validation-skills",
                skills_dir="pattern-validation-skills",
                state_path=repo / ".triton-agent" / "pattern-validation-loop-state.json",
                base_revision="origin/main",
                min_rounds=10,
                max_iterations=5,
                agent_name="codex",
                optimize_knowledge="v1",
            )
        self.assertIn("triton-npu-pattern-validation-loop", prompt)
        self.assertIn("workspace-scaffold-contract.md", prompt)
        self.assertIn("audit_batch.py", prompt)
        self.assertIn("--archive-passed", prompt)
        self.assertIn("_completed/", prompt)
        self.assertIn("optimize-batch", prompt)
        self.assertIn("build_pattern_index.py", prompt)
        self.assertIn("--skills-source-dir", prompt)
        self.assertIn("pattern-validation-skills", prompt)
        self.assertGreaterEqual(prompt.count("--show-output"), 2)
        self.assertIn("omit from optimize-batch (use optimize defaults)", prompt)
        self.assertNotIn("--test-mode", prompt)
        self.assertNotIn("--bench-mode", prompt)
        self.assertNotIn("--target-chip", prompt)

    def test_prompt_includes_optimize_passthrough_flags_when_set(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp)
            synthesis = repo / "PERF_PATTERN_SYNTHESIS.md"
            synthesis.write_text("# Synthesis\n", encoding="utf-8")
            prompt = build_pattern_validation_loop_prompt(
                repo_path=repo,
                synthesis_path=synthesis,
                batch_dir=repo / "pattern-validation-batch",
                skills_workdir=repo / "pattern-validation-skills",
                skills_dir="pattern-validation-skills",
                state_path=repo / ".triton-agent" / "pattern-validation-loop-state.json",
                base_revision="origin/main",
                min_rounds=10,
                max_iterations=5,
                agent_name="opencode",
                optimize_knowledge="v2",
                target_chip="A3",
                test_mode="standalone",
                bench_mode="msprof",
            )
        self.assertIn("--target-chip A3", prompt)
        self.assertIn("--test-mode standalone", prompt)
        self.assertIn("--bench-mode msprof", prompt)
        self.assertIn("target_chip=A3", prompt)
        self.assertNotIn("omit from optimize-batch", prompt)

    def test_build_optimize_batch_extra_flags_omits_unset_values(self) -> None:
        self.assertEqual(build_optimize_batch_extra_flags(), "")
        self.assertEqual(
            build_optimize_batch_extra_flags(test_mode="differential"),
            " --test-mode differential",
        )

    def test_build_request_requires_synthesis_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            with self.assertRaises(ValueError):
                build_pattern_validation_loop_request(target_path=repo)

    @patch("triton_agent.pattern_validation_loop.launcher.resolve_staged_skills")
    @patch("triton_agent.pattern_validation_loop.launcher.seed_pattern_validation_skills_dir")
    def test_build_request_sets_command_kind(
        self,
        mock_seed: unittest.mock.MagicMock,
        mock_staged: unittest.mock.MagicMock,
    ) -> None:
        mock_staged.return_value = (("triton-npu-pattern-validation-loop",), None)
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            (repo / "PERF_PATTERN_SYNTHESIS.md").write_text("# Synthesis\n", encoding="utf-8")
            skills_workdir = repo / "pattern-validation-skills"
            mock_seed.return_value = skills_workdir
            request = build_pattern_validation_loop_request(
                target_path=repo,
                skills_dir="pattern-validation-skills",
            )
        mock_seed.assert_called_once_with(
            repo,
            "pattern-validation-skills",
            optimize_knowledge="v1",
        )
        self.assertEqual(request.command_kind, CommandKind.PATTERN_VALIDATION_LOOP)
        self.assertEqual(request.skill_name, "triton-npu-pattern-validation-loop")
        self.assertIn("pattern-validation-skills", request.prompt)
        self.assertIn("--skills-source-dir", request.prompt)


def _make_git_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "--template=")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "README.md").write_text("init\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo.resolve()


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


if __name__ == "__main__":
    unittest.main()
