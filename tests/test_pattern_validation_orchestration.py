import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_agent.pattern_validation_loop.orchestration import (
    run_pattern_validation_loop_orchestrated,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class PatternValidationOrchestrationTests(unittest.TestCase):
    @patch(
        "triton_agent.pattern_validation_loop.orchestration.generate_workspace_plan_if_present",
        return_value=(None, []),
    )
    @patch("triton_agent.pattern_validation_loop.orchestration.reset_active_workspace_rounds")
    @patch("triton_agent.pattern_validation_loop.orchestration.collect_batch_evidence")
    @patch("triton_agent.pattern_validation_loop.orchestration.run_optimize_batch", return_value=1)
    @patch(
        "triton_agent.pattern_validation_loop.orchestration.run_pattern_validation_verify",
        return_value=0,
    )
    @patch("triton_agent.pattern_validation_loop.orchestration._run_analyze_agent")
    @patch("triton_agent.pattern_validation_loop.orchestration._run_prepare_agent", return_value=0)
    @patch("triton_agent.pattern_validation_loop.orchestration.seed_pattern_validation_skills_dir")
    def test_loop_continues_after_optimize_batch_failure(
        self,
        _mock_seed: unittest.mock.MagicMock,
        mock_prepare: unittest.mock.MagicMock,
        mock_analyze: unittest.mock.MagicMock,
        _mock_verify: unittest.mock.MagicMock,
        mock_optimize: unittest.mock.MagicMock,
        mock_collect: unittest.mock.MagicMock,
        _mock_reset: unittest.mock.MagicMock,
        _mock_plan: unittest.mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            (repo / "PERF_PATTERN_SYNTHESIS.md").write_text("# Synthesis\n", encoding="utf-8")
            state_path = repo / ".triton-agent" / "pattern-validation-loop-state.json"

            def analyze_side_effect(config, **kwargs: object) -> int:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    '{"status": "complete", "iteration": 1, "history": []}\n',
                    encoding="utf-8",
                )
                return 0

            mock_analyze.side_effect = analyze_side_effect

            code = run_pattern_validation_loop_orchestrated(
                target_path=repo,
                max_iterations=3,
                agent_name="codex",
                verbose=False,
                show_output=False,
            )

        self.assertEqual(code, 0)
        mock_collect.assert_called_once()
        mock_analyze.assert_called_once()
        mock_optimize.assert_called_once()
        optimize_options = mock_optimize.call_args[0][1]
        self.assertIn(".py.txt", optimize_options.prompt or "")

    @patch(
        "triton_agent.pattern_validation_loop.orchestration.generate_workspace_plan_if_present",
        return_value=(None, []),
    )
    @patch("triton_agent.pattern_validation_loop.orchestration.reset_active_workspace_rounds")
    @patch("triton_agent.pattern_validation_loop.orchestration.collect_batch_evidence")
    @patch("triton_agent.pattern_validation_loop.orchestration.run_optimize_batch", return_value=0)
    @patch(
        "triton_agent.pattern_validation_loop.orchestration.run_pattern_validation_verify",
        return_value=0,
    )
    @patch("triton_agent.pattern_validation_loop.orchestration._run_analyze_agent")
    @patch("triton_agent.pattern_validation_loop.orchestration._run_prepare_agent", return_value=0)
    @patch("triton_agent.pattern_validation_loop.orchestration.seed_pattern_validation_skills_dir")
    def test_loop_completes_when_analyze_marks_state_complete(
        self,
        _mock_seed: unittest.mock.MagicMock,
        mock_prepare: unittest.mock.MagicMock,
        mock_analyze: unittest.mock.MagicMock,
        _mock_verify: unittest.mock.MagicMock,
        mock_optimize: unittest.mock.MagicMock,
        _mock_collect: unittest.mock.MagicMock,
        _mock_reset: unittest.mock.MagicMock,
        _mock_plan: unittest.mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            (repo / "PERF_PATTERN_SYNTHESIS.md").write_text("# Synthesis\n", encoding="utf-8")
            state_path = repo / ".triton-agent" / "pattern-validation-loop-state.json"

            def analyze_side_effect(config, **kwargs: object) -> int:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    '{"status": "complete", "iteration": 1, "history": []}\n',
                    encoding="utf-8",
                )
                return 0

            mock_analyze.side_effect = analyze_side_effect

            code = run_pattern_validation_loop_orchestrated(
                target_path=repo,
                max_iterations=3,
                agent_name="codex",
                verbose=False,
                show_output=False,
            )

        self.assertEqual(code, 0)
        mock_prepare.assert_called_once()
        mock_optimize.assert_called_once()
        mock_analyze.assert_called_once()
        mock_optimize.assert_called_once()

    @patch(
        "triton_agent.pattern_validation_loop.orchestration.generate_workspace_plan_if_present",
        return_value=(None, []),
    )
    @patch("triton_agent.pattern_validation_loop.orchestration.reset_active_workspace_rounds")
    @patch("triton_agent.pattern_validation_loop.orchestration.collect_batch_evidence")
    @patch("triton_agent.pattern_validation_loop.orchestration.run_optimize_batch", return_value=0)
    @patch(
        "triton_agent.pattern_validation_loop.orchestration.run_pattern_validation_verify",
        return_value=0,
    )
    @patch("triton_agent.pattern_validation_loop.orchestration._run_analyze_agent", return_value=0)
    @patch("triton_agent.pattern_validation_loop.orchestration._run_prepare_agent", return_value=0)
    @patch("triton_agent.pattern_validation_loop.orchestration.seed_pattern_validation_skills_dir")
    def test_loop_resets_and_reoptimizes_when_not_complete(
        self,
        _mock_seed: unittest.mock.MagicMock,
        _mock_prepare: unittest.mock.MagicMock,
        _mock_analyze: unittest.mock.MagicMock,
        _mock_verify: unittest.mock.MagicMock,
        mock_optimize: unittest.mock.MagicMock,
        _mock_collect: unittest.mock.MagicMock,
        mock_reset: unittest.mock.MagicMock,
        _mock_plan: unittest.mock.MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = _make_git_repo(Path(tmp))
            (repo / "PERF_PATTERN_SYNTHESIS.md").write_text("# Synthesis\n", encoding="utf-8")
            state_path = repo / ".triton-agent" / "pattern-validation-loop-state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text('{"status": "running", "iteration": 1, "history": []}\n', encoding="utf-8")

            code = run_pattern_validation_loop_orchestrated(
                target_path=repo,
                max_iterations=2,
                agent_name="codex",
                verbose=False,
                show_output=False,
            )

        self.assertEqual(code, 1)
        self.assertEqual(mock_optimize.call_count, 2)
        mock_reset.assert_called_once()


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
