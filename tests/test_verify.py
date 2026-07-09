import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.models import AgentResult
from triton_agent.verify.core import VerifyOptions, prepare_verify_target, run_verify


class VerifyTests(unittest.TestCase):
    def _write_baseline(self, workspace: Path) -> None:
        baseline_dir = workspace / "baseline"
        baseline_dir.mkdir()
        (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
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
            "# bench-mode: torch-npu-profiler\nprint('bench')\n",
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
                    "bench_mode": "torch-npu-profiler",
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
        (round_dir / "opt_kernel.py").write_text(
            f"print('round {round_number}')\n",
            encoding="utf-8",
        )
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        (round_dir / "opt_kernel_perf.txt").write_text(perf_text, encoding="utf-8")
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": f"opt-round-{round_number}",
                    "parent_round": "baseline",
                    "hypothesis": "faster",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "opt_kernel_perf.txt",
                    "comparison_target_path": "baseline/perf.txt",
                    "effective_metric_source": "kernel",
                    "summary_path": "summary.md",
                    "opt_note_updated": True
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

            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            self.assertEqual(target.selected_round, "round-2")
            self.assertEqual(target.round_dir, best_round)
            self.assertEqual(target.source_operator, best_round / "opt_kernel.py")
            self.assertEqual(target.verify_dir, workspace / "opt-verify" / "verify-20260420-153012")
            self.assertEqual(target.copied_operator, target.verify_dir / "opt_kernel.py")
            self.assertEqual(
                target.copied_operator.read_text(encoding="utf-8"),
                "print('round 2')\n",
            )
            self.assertEqual(target.source_test_file, workspace / "differential_test_kernel.py")
            self.assertEqual(target.test_file, target.verify_dir / "differential_test_kernel.py")
            self.assertEqual(target.source_bench_file, workspace / "bench_kernel.py")
            self.assertEqual(target.bench_file, target.verify_dir / "bench_kernel.py")
            self.assertEqual(target.source_baseline_operator, workspace / "baseline" / "kernel.py")
            self.assertEqual(target.baseline_operator, target.verify_dir / "baseline_kernel.py")
            self.assertEqual(target.test_mode, "differential")
            self.assertEqual(target.bench_mode, "torch-npu-profiler")
            self.assertEqual(target.baseline_perf, workspace / "baseline" / "perf.txt")

    def test_prepare_target_uses_unique_verify_directory_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            existing = workspace / "opt-verify" / "verify-20260420-153012"
            existing.mkdir(parents=True)
            (existing / "opt_kernel.py").write_text("existing\n", encoding="utf-8")

            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            self.assertEqual(target.verify_dir, workspace / "opt-verify" / "verify-20260420-153012-2")
            self.assertEqual((existing / "opt_kernel.py").read_text(encoding="utf-8"), "existing\n")

    def test_prepare_target_accepts_legacy_round_operator_and_perf_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "kernel.py").write_text("print('legacy round')\n", encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("latency-a: 8\nlatency-b: 18\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "baseline",
                        "hypothesis": "faster",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                    "perf_artifact": "perf.txt",
                    "comparison_target_path": "baseline/perf.txt",
                    "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True
                    }
                ),
                encoding="utf-8",
            )
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )

            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            self.assertEqual(target.source_operator, round_dir / "kernel.py")
            self.assertEqual(target.copied_operator, target.verify_dir / "kernel.py")
            self.assertEqual(
                target.copied_operator.read_text(encoding="utf-8"),
                "print('legacy round')\n",
            )

    def test_run_verify_all_uses_copied_operator_and_writes_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )
            baseline_perf_path = target.verify_dir / "baseline_kernel_perf.txt"
            perf_path = target.verify_dir / "opt_kernel_perf.txt"
            result_path = target.verify_dir / "opt_kernel_result.pt"
            baseline_perf_path.write_text("latency-a: 11\nlatency-b: 22\n", encoding="utf-8")
            perf_path.write_text("latency-a: 8\nlatency-b: 18\n", encoding="utf-8")

            with patch(
                "triton_agent.verify.core.run_local_test",
                return_value=(AgentResult(return_code=0, stdout="test ok\n", stderr=""), result_path),
            ) as run_test:
                with patch(
                    "triton_agent.verify.core.run_local_bench",
                    side_effect=[
                        (AgentResult(return_code=0, stdout="baseline bench ok\n", stderr=""), baseline_perf_path),
                        (AgentResult(return_code=0, stdout="bench ok\n", stderr=""), perf_path),
                    ],
                ) as run_bench:
                    with patch(
                        "triton_agent.verify.core.compare_perf_files",
                        return_value=0,
                    ) as compare_perf:
                        result = run_verify(
                            target,
                            VerifyOptions(phase="all"),
                        )

            self.assertEqual(result.return_code, 0)
            run_test.assert_called_once_with(
                target.test_file,
                target.copied_operator,
                target.test_mode,
            )
            self.assertEqual(
                run_bench.call_args_list,
                [
                    ((target.bench_file, target.baseline_operator, target.bench_mode), {}),
                    ((target.bench_file, target.copied_operator, target.bench_mode), {}),
                ],
            )
            compare_perf.assert_called_once_with(
                baseline_perf_path,
                perf_path,
                metric_source="kernel",
            )
            self.assertEqual((target.verify_dir / "test.log").read_text(encoding="utf-8"), "test ok\n")
            self.assertEqual(
                (target.verify_dir / "rerun-best-bench.log").read_text(encoding="utf-8"),
                "bench ok\n",
            )
            self.assertEqual(
                (target.verify_dir / "rerun-baseline-bench.log").read_text(encoding="utf-8"),
                "baseline bench ok\n",
            )

            state = json.loads((target.verify_dir / "verify-state.json").read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(state),
                ["inputs", "selection", "verify-result", "workspace"],
            )
            self.assertEqual(state["selection"]["round_dir"], "opt-round-1")
            self.assertEqual(state["selection"]["source_operator"], "opt-round-1/opt_kernel.py")
            self.assertEqual(state["selection"]["numeric_best_source"], "optimize-status")
            optimize_status = state["selection"]["optimize_status"]
            self.assertEqual(
                sorted(optimize_status),
                [
                    "avg_improvement",
                    "geomean_speedup",
                    "state",
                    "warnings",
                ],
            )
            self.assertEqual(optimize_status["state"], "ok")
            self.assertAlmostEqual(optimize_status["avg_improvement"], 0.15)
            self.assertAlmostEqual(optimize_status["geomean_speedup"], (10 / 8 * 20 / 18) ** 0.5)
            self.assertEqual(optimize_status["warnings"], [])
            self.assertEqual(
                state["workspace"],
                {
                    "verify_dir": "opt-verify/verify-20260420-153012",
                    "operator": "opt-verify/verify-20260420-153012/opt_kernel.py",
                    "baseline_operator": "opt-verify/verify-20260420-153012/baseline_kernel.py",
                },
            )
            self.assertEqual(
                state["inputs"],
                {
                    "test_harness": {
                        "source": "differential_test_kernel.py",
                        "copied": "opt-verify/verify-20260420-153012/differential_test_kernel.py",
                        "mode": "differential",
                    },
                    "bench_harness": {
                        "source": "bench_kernel.py",
                        "copied": "opt-verify/verify-20260420-153012/bench_kernel.py",
                        "mode": "torch-npu-profiler",
                    },
                    "baseline_perf": "baseline/perf.txt",
                },
            )
            verify_result = state["verify-result"]
            self.assertEqual(verify_result["test"]["status"], "passed")
            self.assertEqual(verify_result["test"]["return_code"], 0)
            self.assertEqual(
                verify_result["test"]["result_artifact"],
                "opt-verify/verify-20260420-153012/opt_kernel_result.pt",
            )
            self.assertEqual(
                verify_result["rerun_baseline_bench"]["perf_artifact"],
                "opt-verify/verify-20260420-153012/baseline_kernel_perf.txt",
            )
            self.assertEqual(verify_result["rerun_baseline_bench"]["status"], "passed")
            self.assertEqual(verify_result["rerun_baseline_bench"]["return_code"], 0)
            self.assertEqual(
                verify_result["rerun_baseline_bench"]["latency"],
                {"latency-a": 11.0, "latency-b": 22.0},
            )
            self.assertEqual(
                verify_result["rerun_best_bench"]["perf_artifact"],
                "opt-verify/verify-20260420-153012/opt_kernel_perf.txt",
            )
            self.assertEqual(verify_result["rerun_best_bench"]["status"], "passed")
            self.assertEqual(verify_result["rerun_best_bench"]["return_code"], 0)
            self.assertEqual(
                verify_result["rerun_best_bench"]["latency"],
                {"latency-a": 8.0, "latency-b": 18.0},
            )
            self.assertEqual(verify_result["compare_perf"]["status"], "passed")
            self.assertEqual(verify_result["compare_perf"]["return_code"], 0)
            self.assertEqual(
                sorted(verify_result["speedup"]),
                ["avg_improvement", "geomean_speedup", "warnings"],
            )
            self.assertAlmostEqual(verify_result["speedup"]["avg_improvement"], ((11 - 8) / 11 + (22 - 18) / 22) / 2)
            self.assertAlmostEqual(verify_result["speedup"]["geomean_speedup"], (11 / 8 * 22 / 18) ** 0.5)
            self.assertEqual(verify_result["speedup"]["warnings"], [])
            self.assertEqual(
                verify_result["consistency"],
                {
                    "status": "matched",
                    "geomean_speedup_delta": verify_result["consistency"]["geomean_speedup_delta"],
                    "avg_improvement_delta": verify_result["consistency"]["avg_improvement_delta"],
                },
            )
            self.assertAlmostEqual(
                verify_result["consistency"]["geomean_speedup_delta"],
                ((11 / 8 * 22 / 18) ** 0.5) - ((10 / 8 * 20 / 18) ** 0.5),
            )

    def test_run_verify_test_phase_skips_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            with patch(
                "triton_agent.verify.core.run_local_test",
                return_value=(AgentResult(return_code=0, stdout="", stderr=""), None),
            ) as run_test:
                with patch("triton_agent.verify.core.run_local_bench") as run_bench:
                    with patch("triton_agent.verify.core.compare_perf_files") as compare_perf:
                        result = run_verify(
                            target,
                            VerifyOptions(phase="test"),
                        )

            self.assertEqual(result.return_code, 0)
            run_test.assert_called_once()
            run_bench.assert_not_called()
            compare_perf.assert_not_called()

    def test_run_verify_bench_phase_runs_compare_perf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )
            baseline_perf_path = target.verify_dir / "baseline_kernel_perf.txt"
            perf_path = target.verify_dir / "opt_kernel_perf.txt"

            with patch("triton_agent.verify.core.run_local_test") as run_test:
                with patch(
                    "triton_agent.verify.core.run_local_bench",
                    side_effect=[
                        (AgentResult(return_code=0, stdout="", stderr=""), baseline_perf_path),
                        (AgentResult(return_code=0, stdout="", stderr=""), perf_path),
                    ],
                ) as run_bench:
                    with patch(
                        "triton_agent.verify.core.compare_perf_files",
                        return_value=0,
                    ) as compare_perf:
                        result = run_verify(
                            target,
                            VerifyOptions(phase="bench"),
                        )

            self.assertEqual(result.return_code, 0)
            run_test.assert_not_called()
            self.assertEqual(run_bench.call_count, 2)
            compare_perf.assert_called_once_with(
                baseline_perf_path,
                perf_path,
                metric_source="kernel",
            )

    def test_run_verify_bench_phase_uses_total_op_metric_source_when_best_round_requests_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            round_dir = self._write_round(
                workspace,
                1,
                "\n".join(
                    [
                        "latency-a: 8",
                        '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":8.0}]}',
                        "latency-b: 18",
                        '# raw-op-statistic-b: {"ops":[{"op_type":"OpB","avg_time_us":18.0}]}',
                    ]
                )
                + "\n",
            )
            (workspace / "baseline" / "perf.txt").write_text(
                "\n".join(
                    [
                        "latency-a: 10",
                        '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":10.0}]}',
                        "latency-b: 20",
                        '# raw-op-statistic-b: {"ops":[{"op_type":"OpB","avg_time_us":20.0}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
            payload["effective_metric_source"] = "total-op"
            (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")
            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )
            baseline_perf_path = target.verify_dir / "baseline_kernel_perf.txt"
            perf_path = target.verify_dir / "opt_kernel_perf.txt"

            with patch("triton_agent.verify.core.run_local_test") as run_test:
                with patch(
                    "triton_agent.verify.core.run_local_bench",
                    side_effect=[
                        (AgentResult(return_code=0, stdout="", stderr=""), baseline_perf_path),
                        (AgentResult(return_code=0, stdout="", stderr=""), perf_path),
                    ],
                ) as run_bench:
                    with patch(
                        "triton_agent.verify.core.compare_perf_files",
                        return_value=0,
                    ) as compare_perf:
                        result = run_verify(
                            target,
                            VerifyOptions(phase="bench"),
                        )

            self.assertEqual(result.return_code, 0)
            run_test.assert_not_called()
            self.assertEqual(run_bench.call_count, 2)
            compare_perf.assert_called_once_with(
                baseline_perf_path,
                perf_path,
                metric_source="total-op",
            )

    def test_run_verify_consistency_ignores_avg_improvement_and_uses_wide_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )
            target = replace(
                target,
                optimize_status=replace(
                    target.optimize_status,
                    avg_improvement=-0.2,
                    geomean_speedup=1.05,
                ),
            )
            baseline_perf_path = target.verify_dir / "baseline_kernel_perf.txt"
            perf_path = target.verify_dir / "opt_kernel_perf.txt"
            baseline_perf_path.write_text("latency-a: 9\nlatency-b: 18\n", encoding="utf-8")
            perf_path.write_text("latency-a: 8\nlatency-b: 18\n", encoding="utf-8")

            with patch(
                "triton_agent.verify.core.run_local_bench",
                side_effect=[
                    (AgentResult(return_code=0, stdout="", stderr=""), baseline_perf_path),
                    (AgentResult(return_code=0, stdout="", stderr=""), perf_path),
                ],
            ):
                with patch(
                    "triton_agent.verify.core.compare_perf_files",
                    return_value=0,
                ):
                    run_verify(
                        target,
                        VerifyOptions(phase="bench"),
                    )

            state = json.loads((target.verify_dir / "verify-state.json").read_text(encoding="utf-8"))
            consistency = state["verify-result"]["consistency"]
            self.assertEqual(consistency["status"], "matched")
            self.assertAlmostEqual(consistency["avg_improvement_delta"], ((9 - 8) / 9 + (18 - 18) / 18) / 2 + 0.2)
            self.assertGreater(abs(consistency["avg_improvement_delta"]), 0.2)
            self.assertLessEqual(abs(consistency["geomean_speedup_delta"]), 0.2)

    def test_run_verify_stops_after_failed_test_in_all_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._write_baseline(workspace)
            self._write_round(workspace, 1, "latency-a: 8\nlatency-b: 18\n")
            target = prepare_verify_target(
                workspace,
                timestamp_label="20260420-153012",
            )

            with patch(
                "triton_agent.verify.core.run_local_test",
                return_value=(AgentResult(return_code=1, stdout="", stderr="failed\n"), None),
            ):
                with patch("triton_agent.verify.core.run_local_bench") as run_bench:
                    result = run_verify(
                        target,
                        VerifyOptions(phase="all"),
                    )

            self.assertEqual(result.return_code, 1)
            run_bench.assert_not_called()
            state = json.loads((target.verify_dir / "verify-state.json").read_text(encoding="utf-8"))
            verify_result = state["verify-result"]
            self.assertEqual(verify_result["test"]["return_code"], 1)
            self.assertEqual(verify_result["test"]["status"], "failed")
            self.assertIsNone(verify_result["rerun_baseline_bench"])
            self.assertIsNone(verify_result["rerun_best_bench"])
            self.assertIsNone(verify_result["speedup"]["avg_improvement"])
            self.assertEqual(
                verify_result["speedup"]["warnings"],
                ["missing rerun baseline perf data", "missing rerun best perf data"],
            )
            self.assertEqual(verify_result["consistency"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
