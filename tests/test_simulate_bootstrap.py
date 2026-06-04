import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_agent.pattern_validation_loop.simulate_plan import (
    bootstrap_simulate_batch,
    build_simulate_plan_config,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SimulateBootstrapTests(unittest.TestCase):
    def test_bootstrap_fails_when_empty_batch_and_skip_prepare(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT / "tests") as tmp:
            batch = Path(tmp)
            synth = WORKSPACE_ROOT / "tests/_simulate_bootstrap_synth.md"
            synth.write_text("# synth\n", encoding="utf-8")
            config = build_simulate_plan_config(
                target_path=WORKSPACE_ROOT,
                batch_dir=batch.relative_to(WORKSPACE_ROOT).as_posix(),
                synthesis_output="tests/_simulate_bootstrap_synth.md",
                skip_prepare=True,
            )
            try:
                state = WORKSPACE_ROOT / "tests/_simulate_bootstrap_state.json"
                code = bootstrap_simulate_batch(
                    config,
                    workspace_plan_path=None,
                    simulate_state_path=state,
                )
            finally:
                synth.unlink(missing_ok=True)
                state.unlink(missing_ok=True)
        self.assertEqual(code, 1)

    def test_bootstrap_runs_prepare_when_batch_empty(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT / "tests") as tmp:
            batch = Path(tmp)
            synth = WORKSPACE_ROOT / "tests/_simulate_bootstrap_synth2.md"
            synth.write_text("# synth\n", encoding="utf-8")
            state = WORKSPACE_ROOT / "tests/_simulate_bootstrap_state2.json"
            config = build_simulate_plan_config(
                target_path=WORKSPACE_ROOT,
                batch_dir=batch.relative_to(WORKSPACE_ROOT).as_posix(),
                synthesis_output="tests/_simulate_bootstrap_synth2.md",
            )
            try:
                with patch(
                    "triton_agent.pattern_validation_loop.simulate_plan.run_pattern_validation_prepare_agent",
                    return_value=0,
                ) as prepare_mock, patch(
                    "triton_agent.pattern_validation_loop.simulate_plan.prepare_simulate_batch",
                    return_value=0,
                ):
                    code = bootstrap_simulate_batch(
                        config,
                        workspace_plan_path=batch / "workspace-plan.json",
                        simulate_state_path=state,
                    )
                prepare_mock.assert_called_once()
            finally:
                synth.unlink(missing_ok=True)
                state.unlink(missing_ok=True)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
