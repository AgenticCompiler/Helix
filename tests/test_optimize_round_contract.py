import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.round_contract import inspect_round_artifacts, load_round_state
from triton_agent.skill_loader import load_skill_script_module


class OptimizeRoundContractTests(unittest.TestCase):
    def test_runtime_round_helpers_match_shared_optimize_check_contract(self) -> None:
        module = load_skill_script_module("triton-npu-optimize-check", "optimize_check")

        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
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
                        "comparison_target": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
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
            self.assertIn("canonical_baseline", str(ctx.exception))
            self.assertIn("perf_summary_source", str(ctx.exception))

    def test_inspect_round_artifacts_flags_missing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
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
                        "comparison_target": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
                    }
                ),
                encoding="utf-8",
            )

            result = inspect_round_artifacts(round_dir)

            self.assertIn("missing summary.md", result.issues)

    def test_inspect_round_artifacts_uses_state_declared_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            reports_dir = round_dir / "reports"
            bench_dir = round_dir / "bench"
            reports_dir.mkdir()
            bench_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            declared_summary = reports_dir / "final.md"
            declared_summary.write_text("summary\n", encoding="utf-8")
            declared_perf = bench_dir / "candidate_perf.txt"
            declared_perf.write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "alt_kernel.py").write_text("print('alt')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "bench/candidate_perf.txt",
                        "canonical_baseline": "baseline",
                        "comparison_target": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "reports/final.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
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
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("legacy summary\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("legacy perf\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "bench/candidate_perf.txt",
                        "canonical_baseline": "baseline",
                        "comparison_target": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "reports/final.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
                    }
                ),
                encoding="utf-8",
            )

            result = inspect_round_artifacts(round_dir)

            self.assertIn("missing reports/final.md", result.issues)
            self.assertIn("missing bench/candidate_perf.txt", result.issues)

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
                        "perf_artifact": "perf.txt",
                        "canonical_baseline": "baseline",
                        "comparison_target": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
                        "perf_analysis_path": "perf-analysis.md",
                        "analysis_comparison_sources": ["baseline/profile", "baseline/ir"],
                    }
                ),
                encoding="utf-8",
            )

            state = load_round_state(round_dir)

            self.assertEqual(state.perf_analysis_path, "perf-analysis.md")
            self.assertEqual(
                state.analysis_comparison_sources,
                ("baseline/profile", "baseline/ir"),
            )

    def test_inspect_round_artifacts_uses_declared_perf_analysis_path_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
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
                        "perf_artifact": "perf.txt",
                        "canonical_baseline": "baseline",
                        "comparison_target": "baseline/perf.txt",
                        "perf_summary_source": "compare-perf",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
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
