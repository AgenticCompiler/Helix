from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import load_perf_artifacts_module, load_probe_runner_module


class ProbeClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_probe_runner_module()

    def _classify(
        self,
        geomean: float,
        improved: int,
        regressed: int,
        min_speedup: float,
    ) -> str:
        return self.module.classify_probe_result(
            geomean_speedup=geomean,
            improved_cases=improved,
            regressed_cases=regressed,
            min_case_speedup=min_speedup,
        )

    def test_clear_gain_classifies_likely_gain(self) -> None:
        self.assertEqual(
            self._classify(geomean=1.20, improved=5, regressed=0, min_speedup=1.05),
            "likely_gain",
        )

    def test_geomean_regression_classifies_likely_regression(self) -> None:
        self.assertEqual(
            self._classify(geomean=0.90, improved=0, regressed=5, min_speedup=0.88),
            "likely_regression",
        )

    def test_single_large_regression_classifies_likely_regression(self) -> None:
        self.assertEqual(
            self._classify(geomean=1.05, improved=3, regressed=1, min_speedup=0.80),
            "likely_regression",
        )

    def test_small_positive_movement_is_inconclusive(self) -> None:
        self.assertEqual(
            self._classify(geomean=1.05, improved=3, regressed=0, min_speedup=0.97),
            "inconclusive",
        )

    def test_small_negative_movement_is_inconclusive(self) -> None:
        self.assertEqual(
            self._classify(geomean=0.98, improved=1, regressed=2, min_speedup=0.90),
            "inconclusive",
        )

    def test_case_disagreement_blocks_gain(self) -> None:
        self.assertEqual(
            self._classify(geomean=1.12, improved=2, regressed=3, min_speedup=0.95),
            "inconclusive",
        )

    def test_single_bad_case_blocks_gain_even_with_strong_geomean(self) -> None:
        self.assertEqual(
            self._classify(geomean=1.15, improved=4, regressed=1, min_speedup=0.91),
            "inconclusive",
        )

    def test_gain_at_threshold_boundary(self) -> None:
        self.assertEqual(
            self._classify(geomean=1.10, improved=2, regressed=0, min_speedup=0.95),
            "likely_gain",
        )

    def test_regression_at_geomean_boundary(self) -> None:
        self.assertEqual(
            self._classify(geomean=0.95, improved=0, regressed=2, min_speedup=0.90),
            "likely_regression",
        )

    def test_regression_at_single_case_boundary(self) -> None:
        self.assertEqual(
            self._classify(geomean=1.05, improved=2, regressed=1, min_speedup=0.85),
            "likely_regression",
        )


class ProbePerCaseDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_probe_runner_module()

    def _direction(self, baseline: float, compare: float) -> str:
        return self.module.per_case_direction(baseline, compare)

    def test_clear_improvement(self) -> None:
        self.assertEqual(self._direction(baseline=100.0, compare=90.0), "improved")

    def test_clear_regression(self) -> None:
        self.assertEqual(self._direction(baseline=100.0, compare=120.0), "regressed")

    def test_within_positive_threshold_is_unchanged(self) -> None:
        self.assertEqual(self._direction(baseline=100.0, compare=99.5), "unchanged")

    def test_within_negative_threshold_is_unchanged(self) -> None:
        self.assertEqual(self._direction(baseline=100.0, compare=100.5), "unchanged")

    def test_improvement_boundary_excluded(self) -> None:
        self.assertEqual(self._direction(baseline=101.0, compare=100.0), "unchanged")

    def test_regression_boundary_excluded(self) -> None:
        self.assertEqual(self._direction(baseline=100.0, compare=101.01), "unchanged")


class ProbeComparisonAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_probe_runner_module()

    def test_comparison_clear_gain(self) -> None:
        result = self.module.compute_probe_comparison(
            baseline_values={"a": 100.0, "b": 100.0},
            candidate_values={"a": 90.0, "b": 85.0},
        )
        self.assertEqual(result.classification, "likely_gain")
        self.assertEqual(result.improved_cases, 2)
        self.assertEqual(result.regressed_cases, 0)
        self.assertAlmostEqual(result.geomean_speedup, math.sqrt((100 / 90) * (100 / 85)))
        self.assertAlmostEqual(
            result.avg_improvement_pct,
            ((100 / 90 - 1) * 100 + (100 / 85 - 1) * 100) / 2,
        )

    def test_comparison_clear_regression(self) -> None:
        result = self.module.compute_probe_comparison(
            baseline_values={"a": 100.0},
            candidate_values={"a": 120.0},
        )
        self.assertEqual(result.classification, "likely_regression")
        self.assertEqual(result.regressed_cases, 1)

    def test_comparison_inconclusive_small_gain(self) -> None:
        result = self.module.compute_probe_comparison(
            baseline_values={"a": 100.0},
            candidate_values={"a": 95.0},
        )
        self.assertEqual(result.classification, "inconclusive")

    def test_comparison_non_comparable_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.module.compute_probe_comparison(
                baseline_values={"a": 100.0},
                candidate_values={"b": 100.0},
            )

    def test_comparison_partial_overlap_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.module.compute_probe_comparison(
                baseline_values={"a": 100.0, "b": 100.0},
                candidate_values={"a": 90.0, "c": 50.0},
            )

    def test_comparison_mixed_metric_source_detected(self) -> None:
        result = self.module.compute_probe_comparison(
            baseline_values={"a": 100.0, "b": 100.0},
            candidate_values={"a": 90.0, "b": 90.0},
            comparison_modes={"a": "kernel", "b": "total-op"},
        )
        self.assertEqual(result.metric_source_resolved, "mixed")

    def test_comparison_uniform_metric_source_resolved(self) -> None:
        result = self.module.compute_probe_comparison(
            baseline_values={"a": 100.0, "b": 100.0},
            candidate_values={"a": 90.0, "b": 85.0},
            comparison_modes={"a": "kernel", "b": "kernel"},
        )
        self.assertEqual(result.metric_source_resolved, "kernel")


class ProbeCacheValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_probe_runner_module()

    def _expected(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "schema_version": 1,
            "measurement_profile": "probe",
            "probe_contract": {"name": "fast-probe", "warmup_cap": 1, "repeats_cap": 3},
            "baseline_operator_fingerprint": "sha256:aaa",
            "bench_file_fingerprint": "sha256:bbb",
            "bench_cases_fingerprint": "sha256:ccc",
            "bench_mode": "torch-npu-profiler",
            "remote": None,
            "remote_workdir": None,
            "npu_devices": "0",
        }
        base.update(overrides)
        return base

    def test_missing_sidecar_is_miss(self) -> None:
        valid, reason = self.module.cache_is_valid(None, self._expected(), True)
        self.assertFalse(valid)
        self.assertIn("sidecar", reason)

    def test_missing_perf_file_is_miss(self) -> None:
        valid, _ = self.module.cache_is_valid(self._expected(), self._expected(), False)
        self.assertFalse(valid)

    def test_matching_metadata_is_hit(self) -> None:
        valid, reason = self.module.cache_is_valid(
            self._expected(), self._expected(), True
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "hit")

    def test_mismatched_baseline_fingerprint_is_miss(self) -> None:
        valid, reason = self.module.cache_is_valid(
            self._expected(baseline_operator_fingerprint="sha256:different"),
            self._expected(),
            True,
        )
        self.assertFalse(valid)
        self.assertIn("baseline_operator_fingerprint", reason)

    def test_mismatched_bench_mode_is_miss(self) -> None:
        valid, reason = self.module.cache_is_valid(
            self._expected(bench_mode="msprof"),
            self._expected(),
            True,
        )
        self.assertFalse(valid)
        self.assertIn("bench_mode", reason)

    def test_both_omit_bench_cases_fingerprint_is_hit(self) -> None:
        valid, _ = self.module.cache_is_valid(
            self._expected(bench_cases_fingerprint=None),
            self._expected(bench_cases_fingerprint=None),
            True,
        )
        self.assertTrue(valid)

    def test_one_has_bench_cases_fingerprint_other_omits_is_miss(self) -> None:
        valid, reason = self.module.cache_is_valid(
            self._expected(bench_cases_fingerprint="sha256:ccc"),
            self._expected(bench_cases_fingerprint=None),
            True,
        )
        self.assertFalse(valid)
        self.assertIn("bench_cases_fingerprint", reason)


class ProbeCapsWarningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_probe_runner_module()

    def _run_execute_probe(self, bench_mode: str, remote: str | None) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "bench.py"
            op = root / "op.py"
            base = root / "base.py"
            perf = root / "perf.txt"
            bench.write_text("# bench", encoding="utf-8")
            op.write_text("x", encoding="utf-8")
            base.write_text("y", encoding="utf-8")
            perf.write_text("latency-a: 100.0", encoding="utf-8")
            local_run = self.module._LocalRun(
                payload={"return_code": 0, "stdout": "", "stderr": ""},
                perf_path=perf,
            )

            def fake_compare(
                baseline_perf_path: object,
                candidate_perf_path: object,
                *,
                baseline_read_path: object,
                metric_source: object,
                cache_hit: object,
                mismatch_reason: object,
                verbose: object,
                extra_warnings: list[str],
            ) -> object:
                return self.module.ProbeBenchResult(
                    return_code=0,
                    default_lines=[],
                    verbose_lines=[],
                    warnings=list(extra_warnings),
                    remote_workspace=None,
                )

            orig_dir = os.getcwd()
            os.chdir(tmp)
            try:
                with patch.object(self.module, "_run_one_probe", return_value=local_run), \
                     patch.object(self.module, "_compare_probe_artifacts", side_effect=fake_compare):
                    result = self.module._execute_probe(
                        bench,
                        op,
                        base,
                        bench_mode,
                        metric_source="auto",
                        npu_devices=None,
                        verbose=False,
                        remote=remote,
                        remote_workdir=None,
                        keep_remote_workdir=False,
                        stderr=None,
                    )
            finally:
                os.chdir(orig_dir)
            return list(result.warnings)

    def test_warns_when_bench_mode_not_torch_npu_profiler(self) -> None:
        warnings = self._run_execute_probe("msprof", None)
        self.assertTrue(
            any("caps apply only to torch-npu-profiler" in w for w in warnings),
            f"expected caps warning, got: {warnings}",
        )

    def test_no_warning_when_remote_torch_npu_profiler(self) -> None:
        warnings = self._run_execute_probe("torch-npu-profiler", "user@host")
        self.assertFalse(
            any("caps apply only" in w for w in warnings),
            f"expected no caps warning for remote torch-npu-profiler, got: {warnings}",
        )

    def test_no_warning_when_local_torch_npu_profiler(self) -> None:
        warnings = self._run_execute_probe("torch-npu-profiler", None)
        self.assertFalse(
            any("caps apply only" in w for w in warnings),
            f"expected no caps warning, got: {warnings}",
        )

    def test_no_warning_when_standalone_normalizes_to_torch_npu_profiler(self) -> None:
        warnings = self._run_execute_probe("standalone", None)
        self.assertFalse(
            any("caps apply only" in w for w in warnings),
            f"expected no caps warning for standalone (normalizes to torch-npu-profiler), got: {warnings}",
        )


class ProbeBaselineSnapshotTests(unittest.TestCase):

    def setUp(self) -> None:
        self.module = load_probe_runner_module()

    def test_comparison_uses_per_invocation_baseline_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "bench.py"
            op = root / "op.py"
            base = root / "base.py"
            perf = root / "perf.txt"
            bench.write_text("# bench", encoding="utf-8")
            op.write_text("x", encoding="utf-8")
            base.write_text("y", encoding="utf-8")
            perf.write_text("latency-a: 100.0", encoding="utf-8")
            local_run = self.module._LocalRun(
                payload={"return_code": 0, "stdout": "", "stderr": ""},
                perf_path=perf,
            )
            captured: dict[str, object] = {}

            def capture_compare(
                baseline_perf_path: object,
                candidate_perf_path: object,
                *,
                baseline_read_path: object,
                metric_source: object,
                cache_hit: object,
                mismatch_reason: object,
                verbose: object,
                extra_warnings: list[str],
            ) -> object:
                captured["baseline_read_path"] = baseline_read_path
                captured["baseline_perf_path"] = baseline_perf_path
                return self.module.ProbeBenchResult(
                    return_code=0,
                    default_lines=[],
                    verbose_lines=[],
                    warnings=list(extra_warnings),
                    remote_workspace=None,
                )

            orig_dir = os.getcwd()
            os.chdir(tmp)
            try:
                with patch.object(self.module, "_run_one_probe", return_value=local_run), patch.object(
                    self.module, "_compare_probe_artifacts", side_effect=capture_compare
                ):
                    self.module._execute_probe(
                        bench,
                        op,
                        base,
                        "torch-npu-profiler",
                        metric_source="auto",
                        npu_devices=None,
                        verbose=False,
                        remote=None,
                        remote_workdir=None,
                        keep_remote_workdir=False,
                        stderr=None,
                    )
            finally:
                os.chdir(orig_dir)

            read_path = Path(str(captured["baseline_read_path"]))
            display_path = Path(str(captured["baseline_perf_path"]))
            self.assertNotEqual(read_path.name, display_path.name)
            self.assertTrue(
                read_path.name.startswith("baseline_probe_snapshot."),
                f"expected snapshot path, got: {read_path}",
            )
            self.assertEqual(display_path.name, "baseline_probe_perf.txt")
            self.assertFalse(read_path.exists(), f"snapshot not cleaned up: {read_path}")


class ParsePerfPairValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.perf = load_perf_artifacts_module()

    def _write_perf(self, path: Path, case_label: str, bench_mode: str, kernel_us: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "case_label": case_label,
                    "bench_mode": bench_mode,
                    "kernel_avg_time_us": kernel_us,
                }
            ),
            encoding="utf-8",
        )

    def test_rejects_zero_candidate_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.perf.txt"
            cand = root / "cand.perf.txt"
            self._write_perf(base, "a", "torch-npu-profiler", 100.0)
            self._write_perf(cand, "a", "torch-npu-profiler", 0.0)
            with self.assertRaises(ValueError) as ctx:
                self.perf.parse_perf_pair_for_comparison(base, cand, metric_source="kernel")
            self.assertIn("must be > 0", str(ctx.exception))

    def test_rejects_cross_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.perf.txt"
            cand = root / "cand.perf.txt"
            self._write_perf(base, "a", "torch-npu-profiler", 100.0)
            self._write_perf(cand, "a", "msprof", 90.0)
            with self.assertRaises(ValueError) as ctx:
                self.perf.parse_perf_pair_for_comparison(base, cand, metric_source="kernel")
            self.assertIn("different bench modes", str(ctx.exception))

    def test_accepts_valid_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.perf.txt"
            cand = root / "cand.perf.txt"
            self._write_perf(base, "a", "torch-npu-profiler", 100.0)
            self._write_perf(cand, "a", "torch-npu-profiler", 80.0)
            baseline_values, candidate_values, _ = self.perf.parse_perf_pair_for_comparison(
                base, cand, metric_source="kernel"
            )
            self.assertEqual(baseline_values, {"latency-a": 100.0})
            self.assertEqual(candidate_values, {"latency-a": 80.0})

    def test_auto_pair_normalizes_mixed_sources_per_case_to_total_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.perf.txt"
            cand = root / "cand.perf.txt"
            base.write_text(
                (
                    "latency-a: 22.6848\n"
                    '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":22.6848}]}\n'
                ),
                encoding="utf-8",
            )
            cand.write_text(
                (
                    "latency-a: NA\n"
                    '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":10.7328}]}\n'
                ),
                encoding="utf-8",
            )

            baseline_values, candidate_values, comparison_modes = self.perf.parse_perf_pair_for_comparison(
                base, cand, metric_source="auto"
            )

            self.assertEqual(baseline_values, {"latency-a": 22.6848})
            self.assertEqual(candidate_values, {"latency-a": 10.7328})
            self.assertEqual(comparison_modes, {"latency-a": "total-op"})


class RemoteVerboseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_probe_runner_module()

    def test_remote_verbose_omits_workspace_line(self) -> None:
        payload = {"return_code": 0, "stdout": "", "stderr": "some stderr"}
        lines = self.module._remote_verbose(payload, True)
        self.assertFalse(any("Remote workspace" in line for line in lines))
        self.assertIn("some stderr", lines)

    def test_remote_verbose_empty_without_stderr(self) -> None:
        payload = {"return_code": 0, "stdout": "", "stderr": ""}
        lines = self.module._remote_verbose(payload, True)
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
