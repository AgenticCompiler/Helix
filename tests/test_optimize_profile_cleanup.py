import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize import checks as optimize_checks
from triton_agent.optimize.session_artifacts import OptimizeSessionArtifactsManager


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


def _write_baseline(workdir: Path) -> None:
    baseline_dir = workdir / "baseline"
    baseline_dir.mkdir()
    (baseline_dir / "state.json").write_text(
        json.dumps(
            {
                "baseline_kind": "prepared",
                "source_operator": "../kernel.py",
                "baseline_operator": "opt_kernel.py",
                "test_file": "../differential_test_kernel.py",
                "test_mode": "differential",
                "bench_file": "../bench_kernel.py",
                "bench_mode": "torch-npu-profiler",
                "perf_artifact": "perf.txt",
                "correctness_status": "passed",
                "benchmark_status": "passed",
                "baseline_established": True,
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "opt_kernel.py").write_text(TRITON_ROUND_OPERATOR, encoding="utf-8")
    (baseline_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
    (workdir / "kernel.py").write_text(TRITON_ROUND_OPERATOR, encoding="utf-8")
    (workdir / "differential_test_kernel.py").write_text(
        "# test-mode: differential\nprint('test')\n",
        encoding="utf-8",
    )
    (workdir / "bench_kernel.py").write_text(
        "# bench-mode: torch-npu-profiler\n# kernel: add_kernel\nprint('bench')\n",
        encoding="utf-8",
    )
    (workdir / "opt-note.md").write_text("history\n", encoding="utf-8")


def _write_round(workdir: Path, *, profile_dir: Optional[str] = None) -> Path:
    round_dir = workdir / "opt-round-1"
    round_dir.mkdir()
    (round_dir / "opt_kernel.py").write_text(TRITON_ROUND_OPERATOR, encoding="utf-8")
    (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
    (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
    (round_dir / "opt_kernel_perf.txt").write_text("case0: 0.9\n", encoding="utf-8")
    payload: dict[str, object] = {
        "round": "opt-round-1",
        "parent_round": "baseline",
        "hypothesis": "trim profiler artifacts",
        "evidence_sources": ["benchmark"],
        "correctness_status": "passed",
        "benchmark_status": "passed",
        "perf_artifact": "opt_kernel_perf.txt",
        "comparison_target_path": "../baseline/perf.txt",
        "effective_metric_source": "kernel",
        "summary_path": "summary.md",
        "opt_note_updated": True,
    }
    if profile_dir is not None:
        payload["profile_dir"] = profile_dir
    (round_dir / "round-state.json").write_text(json.dumps(payload), encoding="utf-8")
    return round_dir


def _collect_relative_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


class OptimizeProfileCleanupTests(unittest.TestCase):
    def test_submit_round_cleanup_keeps_only_profile_csv_and_deletes_workspace_prof_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _write_baseline(workdir)
            round_dir = _write_round(workdir, profile_dir="profile")
            profile_dir = round_dir / "profile"
            profile_dir.mkdir()
            (profile_dir / "top.txt").write_text("remove\n", encoding="utf-8")
            prof_dir = profile_dir / "PROF_000001"
            output_dir = prof_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_statistic_000001.csv").write_text("Name,Total Time(us)\n", encoding="utf-8")
            (output_dir / "op_summary_000001.csv").write_text("Name,Count\n", encoding="utf-8")
            (output_dir / "op_summary_000001.json").write_text("{}\n", encoding="utf-8")
            (prof_dir / "host" / "sample.json").parent.mkdir(parents=True)
            (prof_dir / "host" / "sample.json").write_text("{}\n", encoding="utf-8")
            (round_dir / "PROF_round_root").mkdir()
            (workdir / "PROF_workspace_root").mkdir()
            (workdir / "OPPROF_workspace_root").mkdir()

            result = optimize_checks.check_round(round_dir)

            self.assertEqual(result.status, "pass")
            self.assertEqual(
                _collect_relative_files(profile_dir),
                {
                    "PROF_000001/mindstudio_profiler_output/op_statistic_000001.csv",
                    "PROF_000001/mindstudio_profiler_output/op_summary_000001.csv",
                },
            )
            self.assertFalse((round_dir / "PROF_round_root").exists())
            self.assertFalse((workdir / "PROF_workspace_root").exists())
            self.assertFalse((workdir / "OPPROF_workspace_root").exists())

    def test_optimize_session_cleanup_repeats_profile_artifact_cleanup_without_submit_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
            )
            round_dir = workdir / "opt-round-1"
            round_dir.mkdir()
            profile_dir = round_dir / "profile"
            nested_dir = profile_dir / "ASCEND_PROFILER_OUTPUT"
            nested_dir.mkdir(parents=True)
            (nested_dir / "kernel_details.csv").write_text("Kernel Name,Total Time(us)\n", encoding="utf-8")
            (nested_dir / "kernel_details.json").write_text("{}\n", encoding="utf-8")
            (profile_dir / "trace.txt").write_text("remove\n", encoding="utf-8")
            (workdir / "PROF_workspace_root").mkdir()
            (workdir / "OPPROF_workspace_root").mkdir()

            warnings = manager.cleanup_checked_session(state)

            self.assertEqual(warnings, [])
            self.assertEqual(
                _collect_relative_files(profile_dir),
                {"ASCEND_PROFILER_OUTPUT/kernel_details.csv"},
            )
            self.assertFalse((workdir / "PROF_workspace_root").exists())
            self.assertFalse((workdir / "OPPROF_workspace_root").exists())


if __name__ == "__main__":
    unittest.main()
