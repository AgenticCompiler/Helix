import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentRequest, CommandKind


class AgentRequestTests(unittest.TestCase):
    def test_with_prompt_preserves_all_other_fields(self) -> None:
        request = AgentRequest(
            command_kind=CommandKind.OPTIMIZE,
            input_path=Path("/tmp/op.py"),
            operator_path=Path("/tmp/op.py"),
            output_path=Path("/tmp/opt_op.py"),
            test_mode="differential",
            bench_mode="standalone",
            interact=False,
            verbose=True,
            show_output=False,
            force_overwrite=False,
            agent_name="codex",
            skill_name="optimize",
            prompt="original",
            workdir=Path("/tmp"),
            min_rounds=2,
            continue_optimize=True,
            require_analysis=True,
            no_agent_session=True,
            staged_skill_names=("optimize", "optimize-supervisor"),
            optimize_role="worker",
            round_brief_path=Path("/tmp/.triton-agent/round-brief.md"),
            supervisor_report_path=Path("/tmp/.triton-agent/supervisor-report.md"),
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
        self.assertEqual(updated.require_analysis, request.require_analysis)
        self.assertEqual(updated.no_agent_session, request.no_agent_session)
        self.assertEqual(updated.staged_skill_names, request.staged_skill_names)
        self.assertEqual(updated.optimize_role, request.optimize_role)
        self.assertEqual(updated.round_brief_path, request.round_brief_path)
        self.assertEqual(updated.supervisor_report_path, request.supervisor_report_path)


if __name__ == "__main__":
    unittest.main()
