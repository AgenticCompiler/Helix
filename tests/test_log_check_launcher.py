import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.log_check.batch import run_log_check_batch
from triton_agent.log_check.log_check_launcher import (
    _LOG_CHECK_JSON_FILENAME,
    _PATTERN_ANALYSIS_JSON_FILENAME,
    build_log_check_request,
    build_log_check_prompt,
    run_log_check,
)
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.otel_trace import TRACE_PATH_ENV
from triton_agent.show_output_log import show_output_log_path
from triton_agent.skills import SkillLinkSet


def _make_log_check_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "overall": "PASS",
            "failed_checks": "",
            "overview_detail": "All checks passed.",
            "checks": [
                {"id": "check-1", "name": "distinct strategies per round", "result": "pass", "detail": "ok"},
                {"id": "check-2", "name": "strategy novelty beyond patterns", "result": "pass", "detail": "ok"},
                {"id": "check-3", "name": "repeated pattern failures", "result": "pass", "detail": "ok"},
                {"id": "check-4", "name": "no code duplication or regression", "result": "pass", "detail": "ok"},
                {"id": "check-6", "name": "code compiles", "result": "pass", "detail": "ok"},
                {"id": "check-7", "name": "uses NPU-specific ops", "result": "pass", "detail": "ok"},
                {"id": "check-8", "name": "safe memory access", "result": "pass", "detail": "ok"},
                {"id": "check-9", "name": "no silent correctness issues", "result": "pass", "detail": "ok"},
            ],
        }
    )


def _make_pattern_analysis_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "rounds": [
                {
                    "round": "round-1",
                    "patterns": [
                        {"name": "tiling", "evidence": "explicit", "source": "round-1/attempts.md"},
                    ],
                },
            ],
            "summary": {
                "given": [{"name": "tiling", "rounds": [1], "evidence": "explicit"}],
                "new": [],
                "extended": [],
            },
        }
    )


def _dummy_resolve_staged_skills(*args, **kwargs):
    return None, None


class _DummySkillLinkManager:
    def prepare_skills(self, agent_name, workdir, *, skill_names=None, skill_sources=None):
        return SkillLinkSet(created_paths=[])

    def describe_prepare(self, links):
        return []

    def describe_cleanup(self, links):
        return []

    def cleanup(self, links):
        return []


class LogCheckLauncherTests(unittest.TestCase):
    def test_log_check_prompt_uses_staged_patterns_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "LayerNorm"
            target.mkdir()

            prompt = build_log_check_prompt(target_path=target, agent_name="codex")
            staged_patterns = ".codex/skills/triton-npu-optimize-knowledge/references/patterns"

            self.assertIn(staged_patterns, prompt)

    def test_run_log_check_uses_target_workspace_as_agent_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "LayerNorm"
            target.mkdir()
            resolved_target = target.resolve()

            captured: dict[str, AgentRequest] = {}

            class DummyRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    (target / _LOG_CHECK_JSON_FILENAME).write_text(
                        _make_log_check_json(), encoding="utf-8"
                    )
                    (target / _PATTERN_ANALYSIS_JSON_FILENAME).write_text(
                        _make_pattern_analysis_json(), encoding="utf-8"
                    )
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.log_check.log_check_launcher.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.log_check.log_check_launcher.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.log_check.log_check_launcher.create_runner",
                return_value=DummyRunner(),
            ):
                exit_code = run_log_check(target_path=target, agent_name="opencode")

            self.assertEqual(exit_code, 0)
            request = captured["request"]
            self.assertEqual(request.workdir, resolved_target)
            self.assertTrue(request.run_id.startswith("log-check-"))
            self.assertEqual(
                show_output_log_path(request),
                resolved_target / "triton-agent-logs" / request.run_id / "show-output.log",
            )

    def test_build_log_check_request_enables_tool_trace_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "LayerNorm"
            target.mkdir()

            request = build_log_check_request(target_path=target, log_tools=True)

            self.assertTrue(request.log_tools)
            self.assertIsNotNone(request.extra_env)
            assert request.extra_env is not None
            trace_path = Path(request.extra_env[TRACE_PATH_ENV])
            self.assertEqual(trace_path.parent.parent, target.resolve() / "triton-agent-logs")
            self.assertTrue(trace_path.parent.name.startswith("log-check-"))
            self.assertEqual(trace_path.name, "tool-traces.jsonl")

    def test_run_log_check_writes_tool_trace_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "LayerNorm"
            target.mkdir()
            captured: dict[str, AgentRequest] = {}

            class DummyRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    trace_path = Path((request.extra_env or {})[TRACE_PATH_ENV])
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    trace_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "type": "agent_invocation",
                                "phase": "end",
                                "command_kind": "log-check",
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    (target / _LOG_CHECK_JSON_FILENAME).write_text(
                        _make_log_check_json(),
                        encoding="utf-8",
                    )
                    (target / _PATTERN_ANALYSIS_JSON_FILENAME).write_text(
                        _make_pattern_analysis_json(),
                        encoding="utf-8",
                    )
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.log_check.log_check_launcher.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.log_check.log_check_launcher.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.log_check.log_check_launcher.create_runner",
                return_value=DummyRunner(),
            ):
                exit_code = run_log_check(target_path=target, log_tools=True)

            self.assertEqual(exit_code, 0)
            trace_path = Path((captured["request"].extra_env or {})[TRACE_PATH_ENV])
            summary = json.loads((trace_path.parent / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["command_kind"], "log-check")
            self.assertEqual(summary["tool_trace_capability"], "agent_invocation_only")

    def test_run_log_check_batch_uses_each_workspace_as_agent_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "LayerNorm"
            workspace.mkdir()
            (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")

            captured: dict[str, AgentRequest] = {}

            class DummyRunner:
                def run(self, request: AgentRequest) -> AgentResult:
                    captured["request"] = request
                    (request.workdir / _LOG_CHECK_JSON_FILENAME).write_text(
                        _make_log_check_json(),
                        encoding="utf-8",
                    )
                    (request.workdir / _PATTERN_ANALYSIS_JSON_FILENAME).write_text(
                        _make_pattern_analysis_json(),
                        encoding="utf-8",
                    )
                    return AgentResult(return_code=0, stdout="", stderr="")

            with patch(
                "triton_agent.log_check.log_check_launcher.resolve_staged_skills",
                side_effect=_dummy_resolve_staged_skills,
            ), patch(
                "triton_agent.log_check.log_check_launcher.SkillLinkManager",
                return_value=_DummySkillLinkManager(),
            ), patch(
                "triton_agent.log_check.log_check_launcher.create_runner",
                return_value=DummyRunner(),
            ):
                exit_code = run_log_check_batch(root, stdout=StringIO())

            self.assertEqual(exit_code, 0)
            request = captured["request"]
            self.assertEqual(request.workdir, workspace.resolve())
            self.assertTrue((root / "log_check_summary.md").exists())
