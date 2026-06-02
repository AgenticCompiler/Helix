import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize import checks as optimize_checks

TRITON_ROUND_OPERATOR = """\
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=128)
    return out
"""

MULTILINE_TRITON_ROUND_OPERATOR = """\
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[
        grid
    ](
        x,
        y,
        out,
        n_elements,
        BLOCK_SIZE=128,
    )
    return out
"""

PURE_TORCH_ROUND_OPERATOR = """\
import torch


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.add(x, y)
"""


class OptimizeCheckTests(unittest.TestCase):
    def test_optimize_checks_delegate_to_optimize_check_script_module(self) -> None:
        module = SimpleNamespace(
            check_baseline=lambda path: {
                "ok": False,
                "kind": "baseline",
                "decision": "revise-required",
                "issues": ("baseline issue",),
                "summary": f"checked {path.name}",
            },
            check_round=lambda path, **__: SimpleNamespace(
                ok=True,
                kind="round",
                decision="pass",
                issues=(),
                summary=f"checked {path.name}",
            ),
        )
        with patch("triton_agent.optimize.checks.load_skill_script_module", return_value=module) as mocked:
            baseline_result = optimize_checks.check_baseline(Path("/tmp/baseline"))
            round_result = optimize_checks.check_round(Path("/tmp/opt-round-1"))

        self.assertFalse(baseline_result.ok)
        self.assertEqual(baseline_result.kind, "baseline")
        self.assertEqual(baseline_result.decision, "revise-required")
        self.assertEqual(baseline_result.issues, ("baseline issue",))
        self.assertEqual(baseline_result.summary, "checked baseline")
        self.assertTrue(round_result.ok)
        self.assertEqual(round_result.kind, "round")
        self.assertEqual(round_result.decision, "pass")
        self.assertEqual(round_result.summary, "checked opt-round-1")
        mocked.assert_any_call("triton-npu-optimize-check", "optimize_check")

    def test_check_baseline_reports_missing_perf_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_dir = workdir / "baseline"
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
                        "bench_mode": "standalone",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

            result = optimize_checks.check_baseline(baseline_dir)

            self.assertFalse(result.ok)
            self.assertEqual(result.kind, "baseline")
            self.assertEqual(result.decision, "revise-required")
            self.assertIn("missing baseline/perf.txt", result.issues)

    def test_check_round_passes_with_complete_round_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1", round_disposition="continue")

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")
            self.assertEqual(result.issues, ())

    def test_check_round_preserves_pt_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1", round_disposition="continue")
            pt_file = round_dir / "test_result.pt"
            pt_file.write_text("stub\n", encoding="utf-8")

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")
            self.assertTrue(pt_file.exists())

    def test_check_round_deletes_pt_files_when_env_var_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1", round_disposition="continue")
            pt_file = round_dir / "test_result.pt"
            pt_file.write_text("stub\n", encoding="utf-8")

            with patch.dict(os.environ, {"TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES": "1"}, clear=False):
                result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")
            self.assertFalse(pt_file.exists())

    def test_check_round_allows_missing_perf_analysis_when_not_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1", round_disposition="continue")

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)

    def test_check_round_flags_missing_declared_perf_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
                perf_analysis_path="perf-analysis.md",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertFalse(result.ok)
            self.assertEqual(result.decision, "revise-required")
            self.assertIn("missing perf-analysis.md", result.issues)

    def test_check_round_rejects_pure_pytorch_operator_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
                operator_source=PURE_TORCH_ROUND_OPERATOR,
            )

            result = optimize_checks.check_round(round_dir)

            self.assertFalse(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "revise-required")
            self.assertIn(
                "round operator no longer preserves a recognizable Triton kernel launch path",
                result.issues,
            )

    def test_check_round_accepts_multiline_triton_launch_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
                operator_source=MULTILINE_TRITON_ROUND_OPERATOR,
            )

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")

    def test_check_round_accepts_legacy_round_artifact_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir(exist_ok=True)
            (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text(TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
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
                        "comparison_target": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "round_disposition": "continue",
                    }
                ),
                encoding="utf-8",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")

    def test_check_round_accepts_operator_named_baseline_perf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline_operator_named_perf(workdir)
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir(exist_ok=True)
            (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text(TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
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
                        "comparison_target": "baseline/kernel_perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "round_disposition": "continue",
                    }
                ),
                encoding="utf-8",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")

    def test_check_round_accepts_total_op_effective_metric_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
            )
            payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
            payload["effective_metric_source"] = "total-op"
            (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")

    def test_check_round_kernel_target_warns_when_effective_metric_source_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
            )
            payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
            payload["effective_metric_source"] = "mixed"
            (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

            result = optimize_checks.check_round(round_dir, optimize_target="kernel")

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")
            self.assertTrue(
                any(
                    issue.startswith(
                        "kernel optimize target fell back to effective_metric_source=mixed"
                    )
                    for issue in result.issues
                )
            )

    def test_check_round_warns_when_recent_rounds_stagnate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(
                workdir,
                perf_text="latency-a: 10.0\n",
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
                round_perf_text="latency-a: 8.5\n",
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_disposition="continue",
                round_perf_text="latency-a: 8.4\n",
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_disposition="continue",
                round_perf_text="latency-a: 8.3\n",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.decision, "pass")
            self.assertTrue(
                any("optimization may be stagnating in the current direction" in issue for issue in result.issues)
            )
            self.assertTrue(
                any("may be stuck in a local optimum" in issue for issue in result.issues)
            )
            self.assertTrue(
                any("Review earlier rounds and consider resuming from a round before this flat sequence" in issue for issue in result.issues)
            )
            self.assertIn("optimization may be stagnating in the current direction", result.summary)

    def test_check_round_does_not_warn_when_recent_rounds_still_improve_meaningfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(
                workdir,
                perf_text="latency-a: 10.0\n",
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
                round_perf_text="latency-a: 8.5\n",
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_disposition="continue",
                round_perf_text="latency-a: 8.4\n",
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_disposition="continue",
                round_perf_text="latency-a: 7.0\n",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.decision, "pass")
            self.assertFalse(
                any("optimization may be stagnating in the current direction" in issue for issue in result.issues)
            )

    def test_check_round_does_not_warn_when_recent_rounds_mix_metric_bases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_perf = "\n".join(
                [
                    "latency-a: 10.0",
                    '# raw-op-statistic-a: {"ops":[{"op_type":"OpA","avg_time_us":50.0}]}',
                ]
            ) + "\n"
            self._write_baseline(
                workdir,
                perf_text=baseline_perf,
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_disposition="continue",
                round_perf_text=baseline_perf.replace("10.0", "8.5").replace("50.0", "45.0"),
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_disposition="continue",
                round_perf_text=baseline_perf.replace("10.0", "8.4").replace("50.0", "42.0"),
                effective_metric_source="total-op",
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_disposition="continue",
                round_perf_text=baseline_perf.replace("10.0", "8.3").replace("50.0", "41.5"),
            )

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.decision, "pass")
            self.assertFalse(
                any("optimization may be stagnating in the current direction" in issue for issue in result.issues)
            )

    def test_check_round_with_remaining_min_rounds_names_next_round_and_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1", round_disposition="continue")

            result = optimize_checks.check_round(round_dir, min_rounds=2)

            self.assertTrue(result.ok)
            self.assertEqual(result.decision, "pass")
            self.assertEqual(result.next_option, "opt-round-2")
            self.assertIn("Next round: opt-round-2.", result.summary)
            self.assertIn("Do not rush into the next code change.", result.summary)
            self.assertIn(
                "First decide which operator, kernel path, or wrapper bottleneck should anchor the next round.",
                result.summary,
            )
            self.assertIn(
                "Decide whether existing evidence is already sufficient or whether profiling, IR, or compiler-source analysis is needed first.",
                result.summary,
            )
            self.assertIn(
                "Do not use agents or subagents to optimize multiple rounds in parallel.",
                result.summary,
            )
            self.assertIn(
                "Do not treat the next round as a parameter-only tuning sweep.",
                result.summary,
            )

    def test_check_round_warns_when_local_optimum_env_vars_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1", round_disposition="continue")

            with patch.dict(
                os.environ,
                {
                    "TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW": "abc",
                    "TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN": "-1",
                },
                clear=False,
            ):
                result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.decision, "pass")
            self.assertTrue(
                any("invalid TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW='abc'; using default 3" in issue for issue in result.issues)
            )
            self.assertTrue(
                any("invalid TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN='-1'; using default 0.02" in issue for issue in result.issues)
            )

    def _write_baseline_with_perf_text(self, workdir: Path, *, perf_text: str) -> None:
        baseline_dir = workdir / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        (workdir / "kernel.py").write_text("print('source')\n", encoding="utf-8")
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
        (baseline_dir / "perf.txt").write_text(perf_text, encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

    def _write_baseline(self, workdir: Path, *, perf_text: str = "latency-a: 1.0\n") -> None:
        self._write_baseline_with_perf_text(workdir, perf_text=perf_text)

    def _write_baseline_operator_named_perf(self, workdir: Path) -> None:
        baseline_dir = workdir / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        (workdir / "kernel.py").write_text("print('source')\n", encoding="utf-8")
        perf_rel = "baseline/kernel_perf.txt"
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
                    "perf_artifact": perf_rel,
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        (baseline_dir / "kernel_perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

    def _write_round(
        self,
        workdir: Path,
        round_name: str,
        *,
        round_disposition: str,
        perf_analysis_path: Optional[str] = None,
        operator_source: str = TRITON_ROUND_OPERATOR,
        round_perf_text: str = "latency-a: 1.0\n",
        effective_metric_source: str = "kernel",
    ) -> Path:
        round_dir = workdir / round_name
        round_dir.mkdir(exist_ok=True)
        (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
        (round_dir / "opt_kernel.py").write_text(operator_source, encoding="utf-8")
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        (round_dir / "opt_kernel_perf.txt").write_text(round_perf_text, encoding="utf-8")
        payload = {
            "round": round_name,
            "parent_round": "round-0",
            "hypothesis": "vectorize loads",
            "evidence_sources": ["benchmark"],
            "correctness_status": "passed",
            "benchmark_status": "passed",
            "perf_artifact": "opt_kernel_perf.txt",
            "canonical_baseline": "baseline",
            "comparison_target": "baseline/perf.txt",
            "perf_summary_source": "compare-perf",
            "effective_metric_source": effective_metric_source,
            "summary_path": "summary.md",
            "opt_note_updated": True,
            "round_disposition": round_disposition,
        }
        if perf_analysis_path is not None:
            payload["perf_analysis_path"] = perf_analysis_path
        (round_dir / "round-state.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return round_dir


if __name__ == "__main__":
    unittest.main()
