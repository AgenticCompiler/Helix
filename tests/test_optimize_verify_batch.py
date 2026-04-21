import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.cli import main
from triton_agent.optimize.verify import OptimizeVerifyResult
from triton_agent.optimize.verify_batch import run_optimize_verify_batch


class OptimizeVerifyBatchTests(unittest.TestCase):
    def _write_verify_state(
        self,
        workspace: Path,
        verify_name: str,
        *,
        test_status: str = "passed",
        baseline_bench_status: str = "passed",
        best_bench_status: str = "passed",
        compare_status: str = "passed",
    ) -> Path:
        verify_dir = workspace / "opt-verify" / verify_name
        verify_dir.mkdir(parents=True)
        state_path = verify_dir / "verify-state.json"
        state_path.write_text(
            json.dumps(
                {
                    "verify-result": {
                        "test": {"status": test_status},
                        "rerun_baseline_bench": {"status": baseline_bench_status},
                        "rerun_best_bench": {"status": best_bench_status},
                        "compare_perf": {"status": compare_status},
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return state_path

    def test_run_optimize_verify_batch_reuses_latest_verify_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            latest_state = self._write_verify_state(workspace, "verify-20260421-120000")
            stream = StringIO()

            with patch("triton_agent.optimize.verify_batch.prepare_optimize_verify_target") as prepare_target:
                with patch("triton_agent.optimize.verify_batch.run_optimize_verify") as run_verify:
                    exit_code = run_optimize_verify_batch(root, stdout=stream)

            self.assertEqual(exit_code, 0)
            prepare_target.assert_not_called()
            run_verify.assert_not_called()
            self.assertIn("[OK] matmul: reused verify-state.json", stream.getvalue())
            self.assertIn(str(latest_state), stream.getvalue())

    def test_run_optimize_verify_batch_force_verify_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "matmul"
            workspace.mkdir()
            self._write_verify_state(workspace, "verify-20260421-120000")
            stream = StringIO()
            verify_dir = workspace / "opt-verify" / "verify-20260421-130000"
            verify_dir.mkdir(parents=True)
            state_path = verify_dir / "verify-state.json"

            with patch(
                "triton_agent.optimize.verify_batch.prepare_optimize_verify_target",
                return_value=object(),
            ) as prepare_target:
                with patch(
                    "triton_agent.optimize.verify_batch.run_optimize_verify",
                    return_value=OptimizeVerifyResult(
                        return_code=0,
                        verify_dir=verify_dir,
                        state_path=state_path,
                    ),
                ) as run_verify:
                    exit_code = run_optimize_verify_batch(root, force_verify=True, stdout=stream)

            self.assertEqual(exit_code, 0)
            prepare_target.assert_called_once_with(workspace)
            run_verify.assert_called_once()
            self.assertIn("[OK] matmul: verified", stream.getvalue())
            self.assertIn(str(state_path), stream.getvalue())

    def test_run_optimize_verify_batch_skips_non_verifiable_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "broken"
            workspace.mkdir()
            stream = StringIO()

            with patch(
                "triton_agent.optimize.verify_batch.prepare_optimize_verify_target",
                side_effect=ValueError("missing baseline"),
            ) as prepare_target:
                exit_code = run_optimize_verify_batch(root, stdout=stream)

            self.assertEqual(exit_code, 0)
            prepare_target.assert_called_once_with(workspace)
            self.assertIn("[SKIP] broken: missing baseline", stream.getvalue())

    def test_run_optimize_verify_batch_continues_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_workspace = root / "ok"
            bad_workspace = root / "bad"
            ok_workspace.mkdir()
            bad_workspace.mkdir()
            stream = StringIO()

            def prepare_side_effect(workspace: Path) -> Path:
                return workspace

            def run_side_effect(target: Path, options: object) -> OptimizeVerifyResult:
                workspace = target
                if workspace == bad_workspace:
                    return OptimizeVerifyResult(
                        return_code=1,
                        verify_dir=workspace / "opt-verify" / "verify-20260421-120000",
                        state_path=workspace / "opt-verify" / "verify-20260421-120000" / "verify-state.json",
                    )
                return OptimizeVerifyResult(
                    return_code=0,
                    verify_dir=workspace / "opt-verify" / "verify-20260421-120000",
                    state_path=workspace / "opt-verify" / "verify-20260421-120000" / "verify-state.json",
                )

            with patch("triton_agent.optimize.verify_batch.prepare_optimize_verify_target", side_effect=prepare_side_effect):
                with patch("triton_agent.optimize.verify_batch.run_optimize_verify", side_effect=run_side_effect):
                    exit_code = run_optimize_verify_batch(root, force_verify=True, stdout=stream)

            self.assertEqual(exit_code, 1)
            rendered = stream.getvalue()
            self.assertIn("[FAIL] bad: verify exited with return code 1", rendered)
            self.assertIn("[OK] ok: verified", rendered)

    def test_main_optimize_verify_batch_dispatches_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "matmul").mkdir()

            with patch(
                "triton_agent.commands.optimize.run_optimize_verify_batch",
                return_value=0,
            ) as run_batch:
                exit_code = main(["optimize-verify-batch", "-i", str(root), "--force-verify"])

            self.assertEqual(exit_code, 0)
            run_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
