import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.optimize.round_contract import inspect_round_artifacts, load_round_state
from helix.skills.loader import load_skill_script_module

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROUND_CONTRACT_PATH = (
    _REPO_ROOT
    / "skills"
    / "common"
    / "ascend-npu-optimize-state"
    / "references"
    / "round-contract.json"
)
_BASELINE_CONTRACT_PATH = (
    _REPO_ROOT
    / "skills"
    / "common"
    / "ascend-npu-optimize-state"
    / "references"
    / "baseline-contract.json"
)


class OptimizeRoundContractTests(unittest.TestCase):
    def test_contract_files_expose_machine_readable_status_enums(self) -> None:
        round_contract = json.loads(_ROUND_CONTRACT_PATH.read_text(encoding="utf-8"))
        baseline_contract = json.loads(_BASELINE_CONTRACT_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            round_contract["status_enums"]["correctness_status"],
            ["passed", "failed", "not_run"],
        )
        self.assertEqual(
            round_contract["status_enums"]["benchmark_status"],
            ["passed", "failed", "not_run"],
        )
        self.assertEqual(
            baseline_contract["status_enums"]["correctness_status"],
            ["passed", "failed", "not_run"],
        )
        self.assertEqual(
            baseline_contract["status_enums"]["benchmark_status"],
            ["passed", "failed", "not_run"],
        )
        self.assertEqual(
            baseline_contract["baseline_state_fields"]["correctness_status"],
            "record the final baseline correctness result; use `passed` only after correctness succeeds. Allowed values: `passed`, `failed`, `not_run`.",
        )
        self.assertEqual(
            baseline_contract["baseline_state_fields"]["benchmark_status"],
            "record the final baseline benchmark result; use `passed` only after the benchmark succeeds. Allowed values: `passed`, `failed`, `not_run`.",
        )
        self.assertEqual(
            round_contract["round_state_required_fields"]["correctness_status"],
            "record the final correctness result for this round; use `passed` only after the round operator passes the chosen correctness check. Allowed values: `passed`, `failed`, `not_run`.",
        )
        self.assertEqual(
            round_contract["round_state_required_fields"]["benchmark_status"],
            "record the final benchmark result for this round; use `passed` only after the round benchmark succeeds and the round perf artifact is written. Allowed values: `passed`, `failed`, `not_run`.",
        )

    def test_runtime_round_helpers_match_split_round_submit_contract(self) -> None:
        module = load_skill_script_module(
            "ascend-npu-optimize-state",
            "round/check",
        )

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
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

            self.assertEqual(load_round_state(round_dir), module.load_round_state(round_dir))
            self.assertEqual(
                inspect_round_artifacts(round_dir),
                module.inspect_round_artifacts(round_dir),
            )

    def test_load_round_state_requires_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps({"round": "opt-round-1"}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_round_state(round_dir)

            self.assertIn("missing required round-state fields", str(ctx.exception))
            self.assertIn("summary_path", str(ctx.exception))
            self.assertIn("correctness_status", str(ctx.exception))

    def test_load_round_state_requires_benchmark_fields_when_benchmark_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_round_state(round_dir)

            self.assertIn("missing required benchmark round-state fields", str(ctx.exception))
            self.assertIn("perf_artifact", str(ctx.exception))
            self.assertIn("comparison_target_path", str(ctx.exception))
            self.assertIn("effective_metric_source", str(ctx.exception))

    def test_load_round_state_allows_missing_benchmark_fields_when_benchmark_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "failed",
                        "benchmark_status": "not_run",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            state = load_round_state(round_dir)

            self.assertIsNone(state.perf_artifact)
            self.assertIsNone(state.comparison_target_path)
            self.assertIsNone(state.effective_metric_source)

    def test_load_round_state_allows_missing_benchmark_fields_when_benchmark_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "failed",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            state = load_round_state(round_dir)

            self.assertIsNone(state.perf_artifact)
            self.assertIsNone(state.comparison_target_path)
            self.assertIsNone(state.effective_metric_source)

    def test_load_round_state_reads_comparison_target_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "../baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            state = load_round_state(round_dir)

            self.assertEqual(state.comparison_target_path, "../baseline/perf.txt")

    def test_load_round_state_accepts_legacy_comparison_target_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target": "../baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            state = load_round_state(round_dir)

            self.assertEqual(state.comparison_target_path, "../baseline/perf.txt")

    def test_load_round_state_rejects_conflicting_comparison_target_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "../baseline/kernel_perf.txt",
                        "comparison_target": "../baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "comparison_target_path and comparison_target disagree",
            ):
                load_round_state(round_dir)

    def test_inspect_round_artifacts_flags_missing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
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

            result = inspect_round_artifacts(round_dir)

            self.assertIn(
                "summary_path points to a missing file: summary.md (expected summary.md)",
                result.issues,
            )

    def test_inspect_round_artifacts_uses_state_declared_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            reports_dir = round_dir / "reports"
            reports_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            declared_summary = reports_dir / "final.md"
            declared_summary.write_text("summary\n", encoding="utf-8")
            declared_perf = round_dir / "opt_kernel_perf.txt"
            declared_perf.write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "reports/final.md",
                        "opt_note_updated": True
                    }
                ),
                encoding="utf-8",
            )

            result = inspect_round_artifacts(round_dir)

            self.assertEqual(result.summary_path, declared_summary)
            self.assertEqual(result.perf_path, declared_perf)
            self.assertEqual(result.issues, ())

    def test_inspect_round_artifacts_prefers_declared_paths_over_legacy_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("legacy summary\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("legacy perf\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "bench/opt_kernel_perf.txt",
                        "canonical_baseline": "baseline",
                        "comparison_target_path": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "effective_metric_source": "kernel",
                        "summary_path": "reports/final.md",
                        "opt_note_updated": True
                    }
                ),
                encoding="utf-8",
            )

            result = inspect_round_artifacts(round_dir)

            self.assertIn(
                "summary_path must use summary.md (got reports/final.md)",
                result.issues,
            )
            self.assertIn(
                "perf_artifact must use opt_kernel_perf.txt (got bench/opt_kernel_perf.txt)",
                result.issues,
            )

    def test_inspect_round_artifacts_accepts_legacy_round_perf_and_operator_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('legacy operator')\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "perf.txt",
                        "canonical_baseline": "baseline",
                        "comparison_target_path": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True
                    }
                ),
                encoding="utf-8",
            )

            result = inspect_round_artifacts(round_dir)

            self.assertEqual(result.operator_path, round_dir / "kernel.py")
            self.assertEqual(result.perf_path, round_dir / "perf.txt")
            self.assertEqual(result.issues, ())

    def test_load_round_state_accepts_optional_perf_analysis_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark", "profile"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "perf_analysis_path": "perf-analysis.md",
                    }
                ),
                encoding="utf-8",
            )

            state = load_round_state(round_dir)

            self.assertEqual(state.perf_analysis_path, "perf-analysis.md")
            self.assertFalse(hasattr(state, "analysis_comparison_sources"))
            self.assertFalse(hasattr(state, "validated_candidate"))

    def test_load_round_state_accepts_effective_metric_source_total_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "reduce wrapper overhead",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "canonical_baseline": "baseline",
                        "comparison_target_path": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "effective_metric_source": "total-op",
                        "summary_path": "summary.md",
                        "opt_note_updated": True
                    }
                ),
                encoding="utf-8",
            )

            state = load_round_state(round_dir)

            self.assertEqual(state.effective_metric_source, "total-op")

    def test_inspect_round_artifacts_uses_declared_perf_analysis_path_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "perf-analysis.md").write_text("analysis\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark", "profile"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "opt_kernel_perf.txt",
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "perf_analysis_path": "perf-analysis.md",
                    }
                ),
                encoding="utf-8",
            )

            result = inspect_round_artifacts(round_dir)

            self.assertEqual(result.perf_analysis_path, round_dir / "perf-analysis.md")
            self.assertEqual(result.issues, ())


if __name__ == "__main__":
    unittest.main()
