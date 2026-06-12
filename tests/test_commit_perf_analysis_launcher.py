import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.commit_perf_analysis.launcher import (
    build_commit_perf_analysis_prompt,
    build_commit_perf_analysis_request,
    count_commits_in_range,
    run_commit_perf_analysis,
)
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.skills import SkillLinkSet

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class _DummySkillLinkManager:
    def prepare_skills(self, agent_name, workdir, *, skill_names=None, skill_sources=None):
        return SkillLinkSet(created_paths=[])

    def describe_prepare(self, links):
        return []

    def describe_cleanup(self, links):
        return []

    def cleanup(self, links):
        return []


class CommitPerfAnalysisLauncherTests(unittest.TestCase):
    def test_prompt_points_agent_to_skill_helper_and_output_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp)
            prompt = build_commit_perf_analysis_prompt(
                repo_path=repo,
                output_path=repo / "PERF_KNOWLEDGE_BASE.md",
                synthesis_output_path=repo / "PERF_PATTERN_SYNTHESIS.md",
                base_revision="main",
                target_chip="A3",
                include_ir=True,
                agent_name="codex",
            )

        self.assertIn(".codex/skills/triton-npu-analyze-commit-perf/scripts/collect_commit_context.py", prompt)
        self.assertIn(".codex/skills/triton-npu-analyze-commit-perf/scripts/group_commit_context_by_file.py", prompt)
        self.assertIn(".codex/skills/triton-npu-analyze-commit-perf/references/output-contract.md", prompt)
        self.assertIn(".codex/skills/triton-npu-analyze-commit-perf/references/pattern-synthesis-contract.md", prompt)
        self.assertIn("triton-npu-optimize-knowledge/references/pattern_index.md", prompt)
        self.assertIn("incrementally by file", prompt)
        self.assertIn("one file group per round", prompt)
        self.assertIn("PERF_PATTERN_SYNTHESIS.md", prompt)
        self.assertIn(".codex/skills/triton-npu-optimize-knowledge", prompt)
        self.assertIn("Target chip:\n\n  A3", prompt)
        self.assertIn("IR support is enabled", prompt)

    def test_build_request_rejects_empty_commit_range(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            with self.assertRaisesRegex(ValueError, "No commits found"):
                build_commit_perf_analysis_request(
                    target_path=repo,
                    base_revision="HEAD",
                    force=True,
                )

    def test_count_commits_in_range_returns_zero_when_base_equals_head(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            self.assertEqual(count_commits_in_range(repo, "HEAD"), 0)

    def test_build_request_rejects_existing_output_without_force(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo, base = _make_git_repo_with_followup_commit(Path(tmp))
            output = repo / "PERF_KNOWLEDGE_BASE.md"
            output.write_text("old report\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already exists"):
                build_commit_perf_analysis_request(
                    target_path=repo,
                    base_revision=base,
                    output=str(output),
                    force=False,
                )

    def test_run_uses_git_root_as_workdir_and_requires_report_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo, base = _make_git_repo_with_followup_commit(Path(tmp))
            captured: dict[str, AgentRequest] = {}

            class DummyRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    if request.output_path is None:
                        raise AssertionError("missing output path")
                    (repo / "PERF_KNOWLEDGE_BASE.md").write_text("# Performance Knowledge Base\n", encoding="utf-8")
                    request.output_path.write_text("# Performance Pattern Synthesis\n", encoding="utf-8")
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.commit_perf_analysis.launcher.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.commit_perf_analysis.launcher.create_runner",
                return_value=DummyRunner(),
            ):
                exit_code = run_commit_perf_analysis(
                    target_path=repo / "kernel.py",
                    base_revision=base,
                    output="report.md",
                    agent_name="codex",
                    force=False,
                )

            self.assertEqual(exit_code, 0)
            request = captured["request"]
            self.assertEqual(request.command_kind, CommandKind.ANALYZE_COMMIT_PERF)
            self.assertEqual(request.workdir, repo)
            self.assertEqual(request.output_path, repo / "report.md")
            self.assertTrue(request.no_agent_session)


def _make_git_repo(root: Path) -> Path:
    repo, _ = _make_git_repo_with_followup_commit(root)
    return repo


def _make_git_repo_with_followup_commit(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "--template=")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "kernel.py").write_text("def kernel():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "kernel.py")
    _git(repo, "commit", "-m", "initial")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "kernel.py").write_text("def kernel():\n    return 2\n", encoding="utf-8")
    _git(repo, "add", "kernel.py")
    _git(repo, "commit", "-m", "opti(kernel): tune kernel")
    return repo.resolve(), base


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
