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

from helix.optimize import checks as optimize_checks

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
    def test_optimize_checks_delegate_to_optimize_state_script_modules(self) -> None:
        module = SimpleNamespace(
            check_baseline=lambda path: {
                "status": "fail",
                "kind": "baseline",
                "issues": ("baseline issue",),
                "summary": f"checked {path.name}",
            },
            check_round=lambda path, **__: SimpleNamespace(
                status="pass",
                kind="round",
                issues=(),
                summary=f"checked {path.name}",
            ),
            count_completed_round_directories=lambda path: 7 if path.name == "workspace" else 0,
            count_terminal_round_directories=lambda path: 9 if path.name == "workspace" else 0,
        )
        with patch("helix.optimize.checks.load_skill_script_module", return_value=module) as mocked:
            baseline_result = optimize_checks.check_baseline(Path("/tmp/baseline"))
            round_result = optimize_checks.check_round(Path("/tmp/opt-round-1"))
            completed_count = optimize_checks.count_completed_round_directories(Path("/tmp/workspace"))
            terminal_count = optimize_checks.count_terminal_round_directories(Path("/tmp/workspace"))

        self.assertEqual(baseline_result.status, "fail")
        self.assertEqual(baseline_result.kind, "baseline")
        self.assertEqual(baseline_result.issues, ("baseline issue",))
        self.assertEqual(baseline_result.summary, "checked baseline")
        self.assertEqual(round_result.status, "pass")
        self.assertEqual(round_result.kind, "round")
        self.assertEqual(round_result.summary, "checked opt-round-1")
        self.assertEqual(completed_count, 7)
        self.assertEqual(terminal_count, 9)
        mocked.assert_any_call(
            "ascend-npu-optimize-state",
            "baseline/check",
        )
        mocked.assert_any_call(
            "ascend-npu-optimize-state",
            "round/check",
        )

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
                        "bench_mode": "torch-npu-profiler",
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

            self.assertEqual(result.status, "fail")
            self.assertEqual(result.kind, "baseline")
            self.assertIn(
                "perf_artifact points to a missing file: baseline/perf.txt",
                result.issues,
            )

    def test_check_round_passes_with_complete_round_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.issues, ())

    def test_terminal_round_counter_counts_rejected_terminal_rounds_only_when_statuses_are_valid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            self._write_round(workdir, "opt-round-1")
            rejected_round = self._write_round(workdir, "opt-round-2")
            invalid_round = self._write_round(workdir, "opt-round-3")

            rejected_state_path = rejected_round / "round-state.json"
            rejected_state = json.loads(rejected_state_path.read_text(encoding="utf-8"))
            rejected_state["correctness_status"] = "failed"
            rejected_state["benchmark_status"] = "not_run"
            rejected_state.pop("perf_artifact")
            rejected_state.pop("comparison_target_path")
            rejected_state.pop("effective_metric_source")
            rejected_state_path.write_text(
                json.dumps(rejected_state),
                encoding="utf-8",
            )

            invalid_state_path = invalid_round / "round-state.json"
            invalid_state = json.loads(invalid_state_path.read_text(encoding="utf-8"))
            invalid_state["correctness_status"] = "maybe"
            invalid_state["benchmark_status"] = "not_run"
            invalid_state_path.write_text(
                json.dumps(invalid_state),
                encoding="utf-8",
            )

            self.assertEqual(optimize_checks.count_terminal_round_directories(workdir), 2)

    def test_check_round_reports_status_not_missing_perf_when_benchmark_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")

            state_path = round_dir / "round-state.json"
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            payload["correctness_status"] = "failed"
            payload["benchmark_status"] = "not_run"
            payload.pop("perf_artifact")
            payload.pop("comparison_target_path")
            payload.pop("effective_metric_source")
            state_path.write_text(json.dumps(payload), encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").unlink()

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "fail")
            self.assertEqual(result.issues, ("correctness_status=failed",))

    def test_check_round_deletes_pt_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")
            pt_file = round_dir / "test_result.pt"
            pt_file.write_text("stub\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HELIX_OPTIMIZE_DELETE_PT_FILES", None)
                result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")
            self.assertFalse(pt_file.exists())

    def test_check_round_deletes_pt_files_when_round_cleanup_policy_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")
            pt_file = round_dir / "test_result.pt"
            pt_file.write_text("stub\n", encoding="utf-8")

            with patch.dict(os.environ, {"HELIX_OPTIMIZE_DELETE_PT_FILES": "round"}, clear=False):
                result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")
            self.assertFalse(pt_file.exists())

    def test_check_round_deletes_pt_files_for_legacy_truthy_cleanup_values(self) -> None:
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                self._write_baseline(workdir)
                round_dir = self._write_round(workdir, "opt-round-1")
                pt_file = round_dir / "test_result.pt"
                pt_file.write_text("stub\n", encoding="utf-8")

                with patch.dict(
                    os.environ,
                    {"HELIX_OPTIMIZE_DELETE_PT_FILES": value},
                    clear=False,
                ):
                    result = optimize_checks.check_round(round_dir)

                self.assertEqual(result.status, "pass")
                self.assertFalse(pt_file.exists())

    def test_check_round_preserves_pt_files_for_legacy_falsey_cleanup_values(self) -> None:
        for value in ("0", "false", "no", "off"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                self._write_baseline(workdir)
                round_dir = self._write_round(workdir, "opt-round-1")
                pt_file = round_dir / "test_result.pt"
                pt_file.write_text("stub\n", encoding="utf-8")

                with patch.dict(
                    os.environ,
                    {"HELIX_OPTIMIZE_DELETE_PT_FILES": value},
                    clear=False,
                ):
                    result = optimize_checks.check_round(round_dir)

                self.assertEqual(result.status, "pass")
                self.assertTrue(pt_file.exists())

    def test_check_round_deletes_prof_artifacts_after_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")
            profile_dir = round_dir / "PROF_000001"
            profile_dir.mkdir()
            (profile_dir / "trace.json").write_text("{}\n", encoding="utf-8")
            profile_file = round_dir / "PROF_marker"
            profile_file.write_text("profile\n", encoding="utf-8")

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertFalse(profile_dir.exists())
            self.assertFalse(profile_file.exists())

    def test_check_round_allows_missing_perf_analysis_when_not_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")

    def test_check_round_reports_summary_path_field_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (workspace / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "opt_kernel.py").write_text(TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
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
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "fail")
            self.assertIn(
                "summary_path points to a missing file: summary.md (expected summary.md)",
                result.issues,
            )

    def test_check_round_flags_missing_declared_perf_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                perf_analysis_path="perf-analysis.md",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "fail")
            self.assertIn(
                "perf_analysis_path points to a missing file: perf-analysis.md (expected perf-analysis.md)",
                result.issues,
            )

    def test_check_round_reports_missing_comparison_target_path_with_field_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")
            payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
            payload["comparison_target_path"] = "../baseline/missing_perf.txt"
            payload.pop("comparison_target", None)
            (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "fail")
            self.assertTrue(
                any(
                    issue.startswith(
                        "comparison_target_path points to a missing file: ../baseline/missing_perf.txt"
                    )
                    for issue in result.issues
                )
            )
            self.assertTrue(
                any("expected ../baseline/perf.txt" in issue for issue in result.issues),
            )

    def test_check_round_reports_noncanonical_comparison_target_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            (workdir / "baseline" / "other_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
            round_dir = self._write_round(workdir, "opt-round-1")
            payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
            payload["comparison_target_path"] = "../baseline/other_perf.txt"
            payload.pop("comparison_target", None)
            (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "fail")
            self.assertIn(
                "comparison_target_path must point to the canonical baseline perf artifact ../baseline/perf.txt (got ../baseline/other_perf.txt)",
                result.issues,
            )

    def test_check_round_reports_baseline_invalid_reason_when_comparison_target_cannot_be_validated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            baseline_payload = json.loads(
                (workdir / "baseline" / "state.json").read_text(encoding="utf-8")
            )
            baseline_payload.pop("perf_artifact")
            (workdir / "baseline" / "state.json").write_text(
                json.dumps(baseline_payload),
                encoding="utf-8",
            )
            round_dir = self._write_round(workdir, "opt-round-1")

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "fail")
            self.assertTrue(
                any(
                    issue.startswith(
                        "cannot validate comparison_target_path because baseline/state.json is invalid:"
                    )
                    for issue in result.issues
                )
            )

    def test_check_baseline_reports_declared_operator_field_name(self) -> None:
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
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")

            result = optimize_checks.check_baseline(baseline_dir)

            self.assertEqual(result.status, "fail")
            self.assertIn(
                "baseline_operator points to a missing file: baseline/kernel.py",
                result.issues,
            )

    def test_check_round_rejects_pure_pytorch_operator_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                operator_source=PURE_TORCH_ROUND_OPERATOR,
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "fail")
            self.assertEqual(result.kind, "round")
            self.assertIn(
                "round operator no longer preserves a recognizable Ascend kernel launch path",
                result.issues,
            )

    def test_check_round_accepts_multiline_triton_launch_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
                operator_source=MULTILINE_TRITON_ROUND_OPERATOR,
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")

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
                        "comparison_target_path": "baseline/perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")

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
                        "comparison_target_path": "baseline/kernel_perf.txt",
                        "effective_metric_source": "kernel",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                    }
                ),
                encoding="utf-8",
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")

    def test_check_round_accepts_total_op_effective_metric_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
            )
            payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
            payload["effective_metric_source"] = "total-op"
            (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")

    def test_check_round_kernel_target_warns_when_effective_metric_source_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(
                workdir,
                "opt-round-1",
            )
            payload = json.loads((round_dir / "round-state.json").read_text(encoding="utf-8"))
            payload["effective_metric_source"] = "mixed"
            (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")

            result = optimize_checks.check_round(round_dir, optimize_target="kernel")

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")
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
                perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":10.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.5,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.4,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.3,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
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
                perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":10.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.5,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.4,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":7.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertFalse(
                any("optimization may be stagnating in the current direction" in issue for issue in result.issues)
            )

    def test_check_round_does_not_warn_when_recent_window_contains_a_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(
                workdir,
                perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":10.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":5.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.3,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertFalse(
                any("optimization may be stagnating in the current direction" in issue for issue in result.issues)
            )
            self.assertFalse(
                any("may be stuck in a local optimum" in issue for issue in result.issues)
            )

    def test_check_round_warns_when_recent_rounds_stagnate_with_slight_decline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(
                workdir,
                perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":10.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.3,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.4,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":8.5,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertTrue(
                any("may be stuck in a local optimum" in issue for issue in result.issues)
            )
            self.assertTrue(
                any("only marginal baseline-relative geomean speedup changes" in issue for issue in result.issues)
            )
            self.assertFalse(
                any("only marginal baseline-relative geomean speedup gains" in issue for issue in result.issues)
            )

    def test_check_round_warns_when_recent_rounds_hit_symmetric_gain_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(
                workdir,
                perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":100.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":100.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":98.0392156862745,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_perf_text='{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":100.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertTrue(
                any("only marginal baseline-relative geomean speedup changes" in issue for issue in result.issues)
            )
            self.assertTrue(
                any("+0.02x, -0.02x" in issue for issue in result.issues)
            )

    def test_check_round_does_not_warn_when_recent_rounds_mix_metric_bases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_perf = (
                '{"case_label":"a","kernel_names":[],"kernel_source":"fixture",'
                '"kernel_avg_time_us":10.0,"ops":[{"op_type":"OpA","avg_time_us":50.0}],'
                '"total_op_avg_time_us":50.0,"error_message":null,"case_wall_clock_seconds":null}\n'
            )
            self._write_baseline(
                workdir,
                perf_text=baseline_perf,
            )
            self._write_round(
                workdir,
                "opt-round-1",
                round_perf_text=baseline_perf.replace("10.0", "8.5").replace("50.0", "45.0"),
            )
            self._write_round(
                workdir,
                "opt-round-2",
                round_perf_text=baseline_perf.replace("10.0", "8.4").replace("50.0", "42.0"),
                effective_metric_source="total-op",
            )
            round_dir = self._write_round(
                workdir,
                "opt-round-3",
                round_perf_text=baseline_perf.replace("10.0", "8.3").replace("50.0", "41.5"),
            )

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertFalse(
                any("optimization may be stagnating in the current direction" in issue for issue in result.issues)
            )

    def test_check_round_with_remaining_batch_rounds_names_next_round_without_reflection_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-2")

            result = optimize_checks.check_round(round_dir, current_round=2, final_round=4)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.next_option, "opt-round-3")
            self.assertIn("Round 2/4 in the current worker batch is complete.", result.summary)
            self.assertIn("Next round: opt-round-3.", result.summary)
            self.assertIn(
                "Use the staged `ascend-npu-optimize-state` skill's `start-round` subcommand to open opt-round-3 before beginning the next round.",
                result.summary,
            )
            self.assertNotIn("Do not rush into the next code change.", result.summary)
            self.assertNotIn("First decide which operator, kernel path, or wrapper bottleneck", result.summary)
            self.assertNotIn("Decide whether existing evidence is already sufficient", result.summary)
            self.assertNotIn("Do not use agents or subagents to optimize multiple rounds in parallel.", result.summary)
            self.assertNotIn("Do not treat the next round as a parameter-only tuning sweep.", result.summary)

    def test_check_round_final_batch_round_says_batch_target_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-4")

            result = optimize_checks.check_round(round_dir, current_round=4, final_round=4)

            self.assertEqual(result.status, "pass")
            self.assertIsNone(result.next_option)
            self.assertIn("This round satisfied the current worker batch target.", result.summary)

    def test_check_round_warns_when_local_optimum_env_vars_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1")

            with patch.dict(
                os.environ,
                {
                    "HELIX_OPTIMIZE_LOCAL_OPTIMUM_WINDOW": "abc",
                    "HELIX_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN": "-1",
                },
                clear=False,
            ):
                result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            # Invalid env vars silently fall back to defaults without warnings.
            self.assertFalse(
                any("invalid" in issue and "LOCAL_OPTIMUM" in issue for issue in result.issues),
            )

    def test_check_round_accepts_state_relative_baseline_and_comparison_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_dir = workdir / "baseline"
            baseline_dir.mkdir()
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()
            (workdir / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workdir / "differential_test_kernel.py").write_text("print('test')\n", encoding="utf-8")
            (workdir / "bench_kernel.py").write_text("print('bench')\n", encoding="utf-8")
            (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (baseline_dir / "perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
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
            (round_dir / "opt_kernel.py").write_text(TRITON_ROUND_OPERATOR, encoding="utf-8")
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
            (round_dir / "opt_kernel_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
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

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.kind, "round")

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
                    "bench_mode": "torch-npu-profiler",
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

    def _write_baseline(self, workdir: Path, *, perf_text: str = '{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n') -> None:
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
                    "bench_mode": "torch-npu-profiler",
                    "perf_artifact": perf_rel,
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        (baseline_dir / "kernel_perf.txt").write_text('{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n', encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

    def _write_round(
        self,
        workdir: Path,
        round_name: str,
        *,
        perf_analysis_path: Optional[str] = None,
        operator_source: str = TRITON_ROUND_OPERATOR,
        round_perf_text: str = '{"case_label":"a","kernel_names":[],"kernel_source":"fixture","kernel_avg_time_us":1.0,"ops":null,"total_op_avg_time_us":null,"error_message":null,"case_wall_clock_seconds":null}\n',
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
            "comparison_target_path": "baseline/perf.txt",
            "perf_summary_source": "compare-perf",
            "effective_metric_source": effective_metric_source,
            "summary_path": "summary.md",
            "opt_note_updated": True,
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
