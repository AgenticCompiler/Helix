import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentResult
from triton_agent.optimize.verify import OptimizeVerifyOptions, prepare_optimize_verify_target, run_optimize_verify


class OptimizeVerifyTests(unittest.TestCase):
    def _write_baseline(self, workspace: Path) -> None:
        baseline_dir = workspace / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
        (baseline_dir / "perf.txt").write_text(
            "latency-a: 10\nlatency-b: 20\n",
            encoding="utf-8",
        )
        (workspace / "differential_test_kernel.py").write_text(
            "# test-mode: differential\nprint('test')\n",
            encoding="utf-8",
        )
        (workspace / "bench_kernel.py").write_text(
            "# bench-mode: standalone\nprint('bench')\n",
            encoding="utf-8",
        )
        (baseline_dir / "state.json").write_text(
            json.dumps(
                {
                    "baseline_kind": "prepared",
                    "source_operator": "kernel.py",
                    "baseline_operator": "baseline/kernel.py",
                    "test_file": "differential_test_kernel.py",
                    "test_mode": "differential",
                    "bench_file": "bench_kernel.py",
                    "bench_mode": "standalone",
                    "perf_artifact": "baseline/perf.txt",
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )

    def _write_round(self, workspace: Path, round_number: int, perf_text: str) -> Path:
        round_dir = workspace / f"opt-round-{round_number}"
        round_dir.mkdir()
        (round_dir / "kernel.py").write_text(
            f"print('round {round_number}')\n",
            encoding="utf-8",
        )
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        (round_dir / "perf.txt").write_text(perf_text, encoding="utf-8")
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": f"opt-round-{round_number}",
                    "parent_round": "baseline",
                    "hypothesis": "faster",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "perf.txt",
                    "canonical_baseline": "baseline",
                    "comparison_target": "baseline/perf.txt",
                    "perf_summary_source": "compare-perf",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                    "next_recommendation": "stop",
                }
            ),
            encoding="utf-8",
        )
        return round_dir

    def test_prepare_target_selects_numeric_best_round_and_copies_operator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 9\nlatency-b: 19\n")
            best_round = self._write_round(workspace, 2, "latency-a: 6\nlatency-b: 12\n")

            target = prepare_optimize_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            self.assertEqual(target.selected_round, "round-2")
            self.assertEqual(target.round_dir, best_round)
            self.assertEqual(target.source_operator, best_round / "kernel.py")
            self.assertEqual(target.verify_dir, workspace / "opt-verify" / "verify-20260420-153012")
            self.assertEqual(target.copied_operator, target.verify_dir / "kernel.py")
            self.assertEqual(
                target.copied_operator.read_text(encoding="utf-8"),
                "print('round 2')\n",
            )
            self.assertEqual(target.source_test_file, workspace / "differential_test_kernel.py")
            self.assertEqual(target.test_file, target.verify_dir / "differential_test_kernel.py")
            self.assertEqual(target.source_bench_file, workspace / "bench_kernel.py")
            self.assertEqual(target.bench_file, target.verify_dir / "bench_kernel.py")
            self.assertEqual(target.test_mode, "differential")
            self.assertEqual(target.bench_mode, "standalone")
            self.assertEqual(target.baseline_perf, workspace / "baseline" / "perf.txt")

    def test_prepare_target_uses_unique_verify_directory_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            existing = workspace / "opt-verify" / "verify-20260420-153012"
            existing.mkdir(parents=True)
            (existing / "kernel.py").write_text("existing\n", encoding="utf-8")

            target = prepare_optimize_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            self.assertEqual(target.verify_dir, workspace / "opt-verify" / "verify-20260420-153012-2")
            self.assertEqual((existing / "kernel.py").read_text(encoding="utf-8"), "existing\n")

    def test_run_optimize_verify_all_uses_copied_operator_and_writes_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_optimize_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )
            perf_path = target.verify_dir / "kernel_perf.txt"
            result_path = target.verify_dir / "kernel_result.pt"

            with patch(
                "triton_agent.optimize.verify.run_local_test",
                return_value=(AgentResult(return_code=0, stdout="test ok\n", stderr=""), result_path),
            ) as run_test:
                with patch(
                    "triton_agent.optimize.verify.run_local_bench",
                    return_value=(AgentResult(return_code=0, stdout="bench ok\n", stderr=""), perf_path),
                ) as run_bench:
                    with patch(
                        "triton_agent.optimize.verify.compare_perf_files",
                        return_value=0,
                    ) as compare_perf:
                        result = run_optimize_verify(
                            target,
                            OptimizeVerifyOptions(phase="all"),
                        )

            self.assertEqual(result.return_code, 0)
            run_test.assert_called_once_with(
                target.test_file,
                target.copied_operator,
                target.test_mode,
            )
            run_bench.assert_called_once_with(
                target.bench_file,
                target.copied_operator,
                target.bench_mode,
            )
            compare_perf.assert_called_once_with(target.baseline_perf, perf_path)
            self.assertEqual((target.verify_dir / "test.log").read_text(encoding="utf-8"), "test ok\n")
            self.assertEqual((target.verify_dir / "bench.log").read_text(encoding="utf-8"), "bench ok\n")

            state = json.loads((target.verify_dir / "verify-state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["selected_round"], "round-1")
            self.assertEqual(state["copied_operator"], "opt-verify/verify-20260420-153012/kernel.py")
            self.assertEqual(state["test"]["return_code"], 0)
            self.assertEqual(state["bench"]["return_code"], 0)
            self.assertEqual(state["compare_perf"]["return_code"], 0)

    def test_run_optimize_verify_test_phase_skips_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_optimize_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            with patch(
                "triton_agent.optimize.verify.run_local_test",
                return_value=(AgentResult(return_code=0, stdout="", stderr=""), None),
            ) as run_test:
                with patch("triton_agent.optimize.verify.run_local_bench") as run_bench:
                    with patch("triton_agent.optimize.verify.compare_perf_files") as compare_perf:
                        result = run_optimize_verify(
                            target,
                            OptimizeVerifyOptions(phase="test"),
                        )

            self.assertEqual(result.return_code, 0)
            run_test.assert_called_once()
            run_bench.assert_not_called()
            compare_perf.assert_not_called()

    def test_run_optimize_verify_bench_phase_runs_compare_perf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_optimize_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )
            perf_path = target.verify_dir / "kernel_perf.txt"

            with patch("triton_agent.optimize.verify.run_local_test") as run_test:
                with patch(
                    "triton_agent.optimize.verify.run_local_bench",
                    return_value=(AgentResult(return_code=0, stdout="", stderr=""), perf_path),
                ) as run_bench:
                    with patch(
                        "triton_agent.optimize.verify.compare_perf_files",
                        return_value=0,
                    ) as compare_perf:
                        result = run_optimize_verify(
                            target,
                            OptimizeVerifyOptions(phase="bench"),
                        )

            self.assertEqual(result.return_code, 0)
            run_test.assert_not_called()
            run_bench.assert_called_once()
            compare_perf.assert_called_once_with(target.baseline_perf, perf_path)

    def test_run_optimize_verify_stops_after_failed_test_in_all_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_optimize_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            with patch(
                "triton_agent.optimize.verify.run_local_test",
                return_value=(AgentResult(return_code=1, stdout="", stderr="failed\n"), None),
            ):
                with patch("triton_agent.optimize.verify.run_local_bench") as run_bench:
                    result = run_optimize_verify(
                        target,
                        OptimizeVerifyOptions(phase="all"),
                    )

            self.assertEqual(result.return_code, 1)
            run_bench.assert_not_called()
            state = json.loads((target.verify_dir / "verify-state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["test"]["return_code"], 1)
            self.assertIsNone(state["bench"])


if __name__ == "__main__":
    unittest.main()
