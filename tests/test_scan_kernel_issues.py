import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skills.loader import load_skill_script_module


_SCAN_MODULE = load_skill_script_module("triton-npu-optimize", "scan_kernel_issues")
_CHECK_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton"
    / "triton-npu-optimize"
    / "scripts"
    / "check_stage_verdict.py"
)


def _write_verdict(round_dir: Path, determined: str, *, algorithmic="clean", parameterization="clean") -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    verdict = {
        "round": round_dir.name,
        "verdicts": [
            {"stage": "boundary", "verdict": "clean"},
            {"stage": "parallel", "verdict": "clean"},
            {"stage": "memory_access", "verdict": "clean"},
            {"stage": "algorithmic", "verdict": algorithmic},
            {"stage": "pipeline", "verdict": "clean"},
            {"stage": "compile_hints", "verdict": "clean"},
            {"stage": "parameterization", "verdict": parameterization},
        ],
        "determined_stage": determined,
    }
    (round_dir / "stage-verdict.json").write_text(json.dumps(verdict), encoding="utf-8")



def _issue_types(source: str) -> list[str]:
    return [item["issue_type"] for item in _SCAN_MODULE.scan_source(source)]


# ---------------------------------------------------------------------------
# Fixture kernels
# ---------------------------------------------------------------------------

_PERMUTE_CONTIGUOUS = """
import torch
import triton

def forward(x):
    return x.permute(0, 1).contiguous()
"""

_IMPLICIT_TRANSPOSE = """
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, c, M, N, K):
    acc = tl.dot(tl.trans(a), b)
    c[:] = acc
"""

_STATIC_RANGE = """
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, N):
    acc = 0.0
    for k in tl.static_range(N):
        acc += x[k]
    out[0] = acc
"""

_FLAT_1D_DECODE = """
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, M, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    row = offs // N
    col = offs % N
    idx = row * N + col
    out[offs] = x[idx]
"""

_INVALID_NUM_WARPS = """
import triton
kernel = lambda *a, **k: None
def launch(x, out):
    grid = (1,)
    kernel[grid](x, out, num_warps=24)
"""

_MISSING_AUTOTUNE = """
import triton
import triton.language as tl

@triton.jit
def kernel(x, out, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    out[offs] = x[offs]
"""

_WRAPPER_LOOP_PER_LAUNCH = """
import torch
import triton

@triton.jit
def kernel(x, out, t, BLOCK: tl.constexpr):
    pass

def forward(x):
    out = torch.empty_like(x)
    for t in range(x.shape[0]):
        grid = (1,)
        kernel[grid](x, out, t)
    return out
"""

_MANUAL_K_REDUCTION = """
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, c, M, N, K, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in range(K):
        a_k = a[pid * BLOCK : (pid + 1) * BLOCK]
        b_k = b[k]
        acc += a_k * b_k
    c[pid * BLOCK : (pid + 1) * BLOCK] = acc
"""

_CLEAN_KERNEL = """
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK": 128}, num_warps=8, num_stages=3)],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :])
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
"""


class ScanKernelIssuesTests(unittest.TestCase):
    def test_detects_permute_contiguous_materialization(self) -> None:
        self.assertIn("permute_contiguous_materialization", _issue_types(_PERMUTE_CONTIGUOUS))

    def test_detects_implicit_transpose_in_dot(self) -> None:
        self.assertIn("implicit_transpose_in_dot", _issue_types(_IMPLICIT_TRANSPOSE))

    def test_detects_static_range_unroll(self) -> None:
        self.assertIn("static_range_unroll", _issue_types(_STATIC_RANGE))

    def test_detects_flat_1d_index_decode(self) -> None:
        self.assertIn("flat_1d_index_decode", _issue_types(_FLAT_1D_DECODE))

    def test_detects_invalid_num_warps(self) -> None:
        self.assertIn("invalid_num_warps", _issue_types(_INVALID_NUM_WARPS))

    def test_detects_missing_autotune(self) -> None:
        self.assertIn("missing_autotune", _issue_types(_MISSING_AUTOTUNE))

    def test_detects_missing_max_contiguous_and_multiple_of(self) -> None:
        types = _issue_types(_MISSING_AUTOTUNE)
        self.assertIn("missing_max_contiguous", types)
        self.assertIn("missing_multiple_of", types)

    def test_detects_wrapper_loop_per_launch(self) -> None:
        self.assertIn("wrapper_loop_per_launch", _issue_types(_WRAPPER_LOOP_PER_LAUNCH))

    def test_detects_manual_k_reduction(self) -> None:
        self.assertIn("manual_k_reduction", _issue_types(_MANUAL_K_REDUCTION))

    def test_clean_kernel_emits_no_high_severity_findings(self) -> None:
        # The clean kernel has autotune, tl.dot (no tl.trans), tl.range-style
        # loop, power-of-two num_warps, and max_contiguous/multiple_of hints.
        # It should not trigger any high-severity (>=4) findings.
        findings = _SCAN_MODULE.scan_source(_CLEAN_KERNEL)
        high = [f for f in findings if f["severity"] >= 4]
        self.assertEqual(high, [], f"unexpected high-severity findings: {high}")

    def test_scan_handles_syntax_error(self) -> None:
        # Must not raise on invalid Python; returns whatever regex finds.
        result = _SCAN_MODULE.scan_source("def broken(:\n  num_warps=24")
        self.assertIsInstance(result, list)

    def test_scan_returns_dict_shape(self) -> None:
        findings = _SCAN_MODULE.scan_source(_STATIC_RANGE)
        self.assertTrue(findings)
        for item in findings:
            self.assertEqual(
                set(item.keys()),
                {"issue_type", "severity", "location", "description", "suggested_fix"},
            )
            self.assertIsInstance(item["severity"], int)


class CheckStageVerdictTests(unittest.TestCase):
    """End-to-end test of the check_stage_verdict.py skill script (subprocess)."""

    def _run_check(self, workdir: Path, round_name: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(_CHECK_SCRIPT), round_name],
            cwd=workdir,
            capture_output=True,
            text=True,
        )

    def test_pass_when_determined_stage_is_dep_order_first(self) -> None:
        # algorithmic has issues, no prereqs; determined=algorithmic -> PASS
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _write_verdict(workdir / "opt-round-1", "algorithmic", algorithmic="issues")
            result = self._run_check(workdir, "opt-round-1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_fail_when_skipping_to_parameterization(self) -> None:
        # algorithmic has issues but determined=parameterization (skips) -> FAIL
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _write_verdict(
                workdir / "opt-round-1", "parameterization", algorithmic="issues", parameterization="issues"
            )
            result = self._run_check(workdir, "opt-round-1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL", result.stderr)

    def test_pass_when_prereqs_declared_clean(self) -> None:
        # algorithmic clean, parameterization issues, determined=parameterization -> PASS
        # (parameterization's prereqs include compile_hints which depends on algorithmic;
        #  but parameterization's direct prereq is compile_hints, and all intermediate
        #  stages are clean, so the chain resolves)
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            _write_verdict(workdir / "opt-round-1", "parameterization", algorithmic="clean", parameterization="issues")
            result = self._run_check(workdir, "opt-round-1")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fail_when_no_verdict_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "opt-round-1").mkdir()
            result = self._run_check(workdir, "opt-round-1")
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
