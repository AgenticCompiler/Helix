import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, CommandKind


class AgentRequestTests(unittest.TestCase):
    def test_agent_request_no_longer_exposes_optimize_role(self) -> None:
        self.assertNotIn("optimize_role", AgentRequest.__dataclass_fields__)

    def test_round_mode_defaults_to_checked(self) -> None:
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="triton-npu-optimize",
            prompt="original",
            workdir=Path("/tmp"),
        )

        self.assertEqual(request.round_mode, "checked")
        self.assertEqual(request.round_batch_size, 10)

    def test_agent_request_supports_mcp_server_names(self) -> None:
        request = AgentRequest(
            command_kind=CommandKind.GEN_TEST,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/test_op.py"),
            test_mode=None,
            bench_mode=None,
            interact=False,
            verbose=False,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="triton-npu-gen-test",
            prompt="original",
            workdir=Path("/tmp"),
            mcp_servers=("triton-agent-run-eval",),
        )

        self.assertEqual(request.mcp_servers, ("triton-agent-run-eval",))

    def test_with_prompt_preserves_all_other_fields(self) -> None:
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="torch-npu-profiler",
            interact=False,
            verbose=True,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="triton-npu-optimize",
            prompt="original",
            workdir=Path("/tmp"),
            min_rounds=2,
            continue_optimize=True,
            no_agent_session=True,
            round_mode="checked",
            round_batch_size=2,
            staged_skill_names=(
                "triton-npu-optimize",
                "triton-npu-optimize-knowledge",
                "triton-npu-optimize-submit-baseline",
                "triton-npu-optimize-submit-round",
                "triton-npu-optimize-start-round",
                "triton-npu-analyze-round-performance",
            ),
            staged_skill_sources={
                "triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2",
            },
            mcp_servers=("triton-agent-run-eval",),
            supervisor_report_path=Path("/tmp/.triton-agent/supervisor-report.md"),
            target_chip="A3",
            optimize_target="operator",
        )

        updated = request.with_prompt("updated")

        self.assertEqual(updated.prompt, "updated")
        self.assertEqual(updated.command_kind, request.command_kind)
        self.assertEqual(updated.input_path, request.input_path)
        self.assertEqual(updated.operator_path, request.operator_path)
        self.assertEqual(updated.output_path, request.output_path)
        self.assertEqual(updated.test_mode, request.test_mode)
        self.assertEqual(updated.bench_mode, request.bench_mode)
        self.assertEqual(updated.agent_name, request.agent_name)
        self.assertEqual(updated.skill_name, request.skill_name)
        self.assertEqual(updated.min_rounds, request.min_rounds)
        self.assertEqual(updated.continue_optimize, request.continue_optimize)
        self.assertEqual(updated.no_agent_session, request.no_agent_session)
        self.assertEqual(updated.round_mode, request.round_mode)
        self.assertEqual(updated.round_batch_size, request.round_batch_size)
        self.assertEqual(updated.staged_skill_names, request.staged_skill_names)
        self.assertEqual(updated.staged_skill_sources, request.staged_skill_sources)
        self.assertEqual(updated.mcp_servers, request.mcp_servers)
        self.assertEqual(updated.supervisor_report_path, request.supervisor_report_path)
        self.assertEqual(updated.target_chip, request.target_chip)
        self.assertEqual(updated.optimize_target, request.optimize_target)


if __name__ == "__main__":
    unittest.main()
