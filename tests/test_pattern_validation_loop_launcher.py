import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_agent.models import CommandKind
from triton_agent.pattern_validation_loop.launcher import (
    OPTIMIZE_BATCH_ENV_PREFIX,
    build_optimize_batch_extra_flags,
    build_optimize_batch_shell_command,
    build_pattern_validation_loop_prompt,
    build_pattern_validation_loop_request,
)
from triton_agent.pattern_validation_loop.prompts import (
    build_analyze_prompt,
    build_prepare_prompt,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class PatternValidationLoopLauncherTests(unittest.TestCase):
    def test_prepare_prompt_references_verify_cli_and_no_optimize_batch(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp)
            synthesis = repo / "PERF_PATTERN_SYNTHESIS.md"
            synthesis.write_text("# Synthesis\n", encoding="utf-8")
            skill_root = repo / ".codex" / "skills" / "triton-npu-pattern-validation-loop"
            skill_root.mkdir(parents=True)
            (skill_root / "references").mkdir(parents=True, exist_ok=True)
            for name in (
                "iteration-contract.md",
                "skill-update-contract.md",
                "workspace-scaffold-contract.md",
            ):
                (skill_root / "references" / name).write_text(f"# {name}\n", encoding="utf-8")
            (skill_root / "SKILL.md").write_text("# skill\n", encoding="utf-8")
            prompt = build_prepare_prompt(
                repo_path=repo,
                synthesis_path=synthesis,
                knowledge_path=repo / "PERF_KNOWLEDGE_BASE.md",
                batch_dir=repo / "pattern-validation-batch",
                workspace_plan_path=repo / "pattern-validation-batch" / "workspace-plan.json",
                skills_workdir=repo / "pattern-validation-skills",
                skills_dir="pattern-validation-skills",
                state_path=repo / ".triton-agent" / "pattern-validation-loop-state.json",
                base_revision="origin/main",
                skill_root=skill_root,
                knowledge_root=repo / "pattern-validation-skills" / "triton-npu-optimize-knowledge",
            )
        self.assertIn("triton-agent pattern-validation-plan", prompt)
        self.assertIn("triton-agent pattern-validation-verify", prompt)
        self.assertIn("workspace-plan.json", prompt)
        self.assertIn("Do not run `triton-agent optimize-batch`", prompt)
        self.assertIn("workspace-scaffold-contract.md", prompt)

    def test_analyze_prompt_requires_simulate_vs_optimize_code_review(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp)
            skill_root = repo / ".codex" / "skills" / "triton-npu-pattern-validation-loop"
            skill_root.mkdir(parents=True)
            (skill_root / "references").mkdir(parents=True, exist_ok=True)
            (skill_root / "references" / "iteration-contract.md").write_text(
                "# iteration\n", encoding="utf-8"
            )
            (skill_root / "SKILL.md").write_text("# skill\n", encoding="utf-8")
            batch = repo / "pattern-validation-batch"
            batch.mkdir()
            prompt = build_analyze_prompt(
                repo_path=repo,
                batch_dir=batch,
                skills_workdir=repo / "pattern-validation-skills",
                state_path=repo / ".triton-agent" / "pattern-validation-loop-state.json",
                audit_report_path=batch / "audit-report.json",
                iteration=1,
                max_iterations=5,
                skill_root=skill_root,
                knowledge_root=repo / "pattern-validation-skills" / "triton-npu-optimize-knowledge",
            )
        self.assertIn("simulate-plan/report.json", prompt)
        self.assertIn("proposed_code_changes.unified_diff", prompt)
        self.assertIn("code_change_alignment", prompt)
        self.assertIn("Code change review", prompt)
        self.assertIn("batch-evaluation.json", prompt)

    def test_legacy_combined_prompt_still_documents_phases(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp)
            synthesis = repo / "PERF_PATTERN_SYNTHESIS.md"
            synthesis.write_text("# Synthesis\n", encoding="utf-8")
            (repo / ".codex" / "skills" / "triton-npu-pattern-validation-loop").mkdir(parents=True)
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
        self.assertIn("pattern-validation-verify", prompt)
        self.assertIn("audit-report.json", prompt)
        self.assertIn("CLI runs optimize-batch", prompt)

    def test_build_optimize_batch_shell_command_includes_env_prefix(self) -> None:
        self.assertEqual(OPTIMIZE_BATCH_ENV_PREFIX, "TRITON_AGENT_STALL_TIMEOUT_SECONDS=0 ")
        command = build_optimize_batch_shell_command(
            batch_dir="/tmp/batch",
            skills_dir="pattern-validation-skills",
            min_rounds=10,
            optimize_knowledge="v1",
            agent_name="opencode",
            resume="fresh",
            reset_optimize=True,
        )
        self.assertTrue(command.startswith("TRITON_AGENT_STALL_TIMEOUT_SECONDS=0 triton-agent optimize-batch"))
        self.assertIn("--reset-optimize", command)

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
        self.assertEqual(request.command_kind, CommandKind.PATTERN_VALIDATION_LOOP)
        self.assertIn("pattern-validation-verify", request.prompt)


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
