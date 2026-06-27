import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.baseline import (
    baseline_gate_issues,
    inspect_baseline_artifacts,
    load_baseline_state,
)
from triton_agent.skill_loader import load_skill_script_module


class OptimizeBaselineTests(unittest.TestCase):
    def test_runtime_baseline_helpers_match_split_baseline_submit_contract(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-submit-baseline",
            "optimize_submit_baseline",
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
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
            (baseline_dir / "perf.txt").write_text("latency-0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

            self.assertEqual(load_baseline_state(workspace), module.load_baseline_state(workspace))
            self.assertEqual(
                inspect_baseline_artifacts(workspace),
                module.inspect_baseline_artifacts(workspace),
            )
            self.assertEqual(baseline_gate_issues(workspace), module.baseline_gate_issues(workspace))

    def test_load_baseline_state_requires_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps({"baseline_kind": "original"}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_baseline_state(workspace)

            self.assertIn("missing required baseline-state fields", str(ctx.exception))
            self.assertIn("baseline_operator", str(ctx.exception))

    def test_inspect_baseline_artifacts_finds_state_perf_and_operator_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
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
            (baseline_dir / "perf.txt").write_text("latency-0: 1.0\n", encoding="utf-8")
            operator_path = baseline_dir / "kernel.py"
            operator_path.write_text("print('baseline')\n", encoding="utf-8")

            inspection = inspect_baseline_artifacts(workspace)

            self.assertEqual(inspection.state_path, baseline_dir / "state.json")
            self.assertEqual(inspection.perf_path, baseline_dir / "perf.txt")
            self.assertEqual(inspection.operator_path, operator_path)
            self.assertEqual(inspection.issues, ())

    def test_inspect_baseline_artifacts_flags_missing_perf_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text("{}", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

            inspection = inspect_baseline_artifacts(workspace)

            self.assertIn("missing baseline/perf.txt", inspection.issues)

    def test_inspect_baseline_artifacts_uses_state_declared_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            metrics_dir = baseline_dir / "metrics"
            snapshots_dir = baseline_dir / "snapshots"
            metrics_dir.mkdir(parents=True)
            snapshots_dir.mkdir(parents=True)
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/snapshots/chosen.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/metrics/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (metrics_dir / "perf.txt").write_text("latency-0: 1.0\n", encoding="utf-8")
            chosen_operator = snapshots_dir / "chosen.py"
            chosen_operator.write_text("print('baseline')\n", encoding="utf-8")
            (baseline_dir / "alt.py").write_text("print('alt')\n", encoding="utf-8")

            inspection = inspect_baseline_artifacts(workspace)

            self.assertEqual(inspection.perf_path, metrics_dir / "perf.txt")
            self.assertEqual(inspection.operator_path, chosen_operator)
            self.assertEqual(inspection.issues, ())

    def test_baseline_gate_issues_reuses_shared_status_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
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
                        "baseline_established": False,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

            issues = baseline_gate_issues(workspace)

            self.assertEqual(issues, ("baseline/state.json marks baseline as not established",))

    def test_inspect_baseline_artifacts_prefers_declared_paths_over_legacy_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/snapshots/chosen.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/metrics/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("legacy perf\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('legacy')\n", encoding="utf-8")

            inspection = inspect_baseline_artifacts(workspace)

            self.assertIn("missing baseline/metrics/perf.txt", inspection.issues)
            self.assertIn("missing baseline/snapshots/chosen.py", inspection.issues)

    def test_inspect_baseline_artifacts_resolves_paths_relative_to_state_file_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "differential_test_kernel.py").write_text("print('test')\n", encoding="utf-8")
            (workspace / "bench_kernel.py").write_text("print('bench')\n", encoding="utf-8")
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "standalone",
                        "perf_artifact": "perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-0: 1.0\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

            inspection = inspect_baseline_artifacts(workspace)

            self.assertEqual(inspection.perf_path, baseline_dir / "perf.txt")
            self.assertEqual(inspection.operator_path, baseline_dir / "kernel.py")
            self.assertEqual(inspection.issues, ())


if __name__ == "__main__":
    unittest.main()
