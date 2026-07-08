"""Code-structure scanner for the optimize orchestrator.

Pure-Python AST/regex scan of a Triton kernel file that emits a list of
mechanical antipattern issues as raw dicts. The runtime loads this skill script
via ``triton_agent.skill_loader.load_skill_script_module`` and coerces each dict
into a typed ``Issue`` via ``triton_agent.optimize.issue_detection.validate_and_coerce``.

This script MUST NOT import ``triton_agent`` (it is a skill-side helper). It uses
only the standard library so it can run under the file-scoped pyright check.

Only mechanical, high-precision detections live here. Semantic detections
(algebraic simplification, fusion opportunities, UB pressure, etc.) are left to
the agent, which writes ``opt-round-N/issues.json`` using the same issue_type
enum.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class _Finding:
    issue_type: str
    severity: int
    location: str
    description: str
    suggested_fix: str


def scan(kernel_path: str | Path) -> List[dict]:
    """Scan a kernel file and return raw issue dicts."""
    path = Path(kernel_path)
    source = path.read_text(encoding="utf-8")
    return scan_source(source)


def scan_source(source: str) -> List[dict]:
    """Scan kernel source text and return raw issue dicts."""
    findings: List[_Finding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None  # fall back to regex-only detection

    for detector in _DETECTORS:
        try:
            findings.extend(detector(source, tree))
        except Exception:
            # A single detector must never abort the whole scan.
            continue

    findings = _dedup(findings)
    return [
        {
            "issue_type": f.issue_type,
            "severity": f.severity,
            "location": f.location,
            "description": f.description,
            "suggested_fix": f.suggested_fix,
        }
        for f in findings
    ]


def _dedup(findings: List[_Finding]) -> List[_Finding]:
    seen: set[Tuple[str, str]] = set()
    unique: List[_Finding] = []
    for f in findings:
        key = (f.issue_type, f.location)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


# ---------------------------------------------------------------------------
# Regex-based detectors
# ---------------------------------------------------------------------------

_PERMUTE_CONTIGUOUS_RE = re.compile(
    r"\.(?:permute|transpose|movedim|T)\([^)]*\)\.contiguous\(\)"
)


def _detect_permute_contiguous_materialization(
    source: str, tree: ast.Module | None
) -> List[_Finding]:
    findings: List[_Finding] = []
    for match in _PERMUTE_CONTIGUOUS_RE.finditer(source):
        line_no = source.count("\n", 0, match.start()) + 1
        findings.append(
            _Finding(
                issue_type="permute_contiguous_materialization",
                severity=5,
                location=f"line {line_no}",
                description=(
                    "Layout materialization via permute/transpose/movedim/T "
                    "followed by .contiguous() before the kernel."
                ),
                suggested_fix=(
                    "Operate on the original strided layout directly with a "
                    "tiled/strided kernel; do not materialize the transpose."
                ),
            )
        )
    return findings


def _detect_implicit_transpose_in_dot(
    source: str, tree: ast.Module | None
) -> List[_Finding]:
    if "tl.dot(" in source and "tl.trans(" in source:
        return [
            _Finding(
                issue_type="implicit_transpose_in_dot",
                severity=4,
                location="tl.dot operand",
                description=(
                    "tl.trans() feeds tl.dot(); the operand is stored in a "
                    "layout that forces a compiler-injected transpose."
                ),
                suggested_fix=(
                    "Align the operand's physical storage layout with what "
                    "tl.dot expects so tl.trans() is unnecessary."
                ),
            )
        ]
    return []


def _detect_static_range_unroll(source: str, tree: ast.Module | None) -> List[_Finding]:
    findings: List[_Finding] = []
    for match in re.finditer(r"tl\.static_range\(", source):
        line_no = source.count("\n", 0, match.start()) + 1
        findings.append(
            _Finding(
                issue_type="static_range_unroll",
                severity=4,
                location=f"line {line_no}",
                description=(
                    "tl.static_range forces full unrolling, which destroys "
                    "software-pipelining overlap across iterations."
                ),
                suggested_fix=(
                    "Replace tl.static_range with tl.range so the compiler "
                    "can preserve loop structure and apply software pipelining."
                ),
            )
        )
    return findings


def _detect_flat_1d_index_decode(source: str, tree: ast.Module | None) -> List[_Finding]:
    if tree is None:
        return []
    has_arange = "tl.arange" in source
    has_pid = "tl.program_id" in source or "pid" in source
    # Coordinate recovery from a flat 1D index uses BOTH floor-division and
    # modulo together (row = idx // N; col = idx % N). A lone % is typically
    # boundary masking and is not a decode signal, so require both.
    has_floor_div = "//" in source
    has_mod = "%" in source
    if not (has_arange and has_pid and has_floor_div and has_mod):
        return []
    # Confirm both // and % appear inside a @triton.jit kernel.
    has_floor_div_in_kernel = False
    has_mod_in_kernel = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FloorDiv) and _inside_jit_kernel(tree, node):
            has_floor_div_in_kernel = True
        if isinstance(node, ast.Mod) and _inside_jit_kernel(tree, node):
            has_mod_in_kernel = True
    if not (has_floor_div_in_kernel and has_mod_in_kernel):
        return []
    return [
        _Finding(
            issue_type="flat_1d_index_decode",
            severity=4,
            location="jit kernel",
            description=(
                "Kernel launches over a flat 1D index (numel) and "
                "recovers multi-dimensional coordinates with // and % "
                "on every lane, dominating SCALAR work."
            ),
            suggested_fix=(
                "Launch over a multi-dimensional grid and index each "
                "axis directly with program_id, eliminating the "
                "per-lane div/mod coordinate decode."
            ),
        )
    ]


def _detect_invalid_num_warps(source: str, tree: ast.Module | None) -> List[_Finding]:
    findings: List[_Finding] = []
    for match in re.finditer(r"num_warps\s*=\s*(\d+)", source):
        value = int(match.group(1))
        if value <= 0 or (value & (value - 1)) != 0:
            line_no = source.count("\n", 0, match.start()) + 1
            findings.append(
                _Finding(
                    issue_type="invalid_num_warps",
                    severity=5,
                    location=f"line {line_no}",
                    description=f"num_warps={value} is not a power of two.",
                    suggested_fix=(
                        "Use a power-of-two num_warps (1, 2, 4, 8, 16, 32)."
                    ),
                )
            )
    return findings


def _detect_missing_autotune(source: str, tree: ast.Module | None) -> List[_Finding]:
    has_jit = "@triton.jit" in source or "triton.jit" in source
    has_autotune = "@triton.autotune" in source or "triton.autotune" in source
    if has_jit and not has_autotune:
        return [
            _Finding(
                issue_type="missing_autotune",
                severity=3,
                location="@triton.jit kernel",
                description=(
                    "Kernel uses @triton.jit but has no @triton.autotune; "
                    "tile/warp/stage parameters are hardcoded."
                ),
                suggested_fix=(
                    "Once the kernel structure is settled, add a "
                    "@triton.autotune configuration grid keyed on shape args."
                ),
            )
        ]
    return []


def _detect_missing_compile_hints(source: str, tree: ast.Module | None) -> List[_Finding]:
    if "@triton.jit" not in source:
        return []
    findings: List[_Finding] = []
    if "max_contiguous" not in source:
        findings.append(
            _Finding(
                issue_type="missing_max_contiguous",
                severity=2,
                location="jit kernel",
                description=(
                    "No tl.max_contiguous hints; the compiler cannot infer "
                    "contiguity facts that would enable wider DMA."
                ),
                suggested_fix=(
                    "After structure is solid, add tl.max_contiguous on "
                    "pointers whose innermost stride is 1."
                ),
            )
        )
    if "multiple_of" not in source:
        findings.append(
            _Finding(
                issue_type="missing_multiple_of",
                severity=2,
                location="jit kernel",
                description=(
                    "No tl.multiple_of hints; the compiler cannot infer "
                    "alignment facts used for vectorized loads/stores."
                ),
                suggested_fix=(
                    "After structure is solid, add tl.multiple_of on "
                    "indices that are known multiples of a power of two."
                ),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# AST-based detectors
# ---------------------------------------------------------------------------


def _detect_wrapper_loop_per_launch(
    source: str, tree: ast.Module | None
) -> List[_Finding]:
    if tree is None:
        return []
    findings: List[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        # A per-launch loop calls something like kernel[grid](...) inside its
        # body: the call target is a Subscript (kernel[grid]).
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Subscript)
            ):
                start_line = node.lineno
                findings.append(
                    _Finding(
                        issue_type="wrapper_loop_per_launch",
                        severity=5,
                        location=f"line {start_line}",
                        description=(
                            "A Python for-loop in the wrapper launches one "
                            "Triton kernel per iteration; per-program launch "
                            "overhead dominates."
                        ),
                        suggested_fix=(
                            "Fuse the loop inside the @triton.jit kernel so "
                            "state stays in registers across iterations."
                        ),
                    )
                )
                break
    return findings


def _detect_manual_k_reduction(source: str, tree: ast.Module | None) -> List[_Finding]:
    if tree is None:
        return []
    findings: List[_Finding] = []
    # A manual K-reduction: a @triton.jit kernel with a for-loop whose body has
    # an AugAssign += accumulating a multiply (acc += a[...] * b[...]).
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not _is_jit_kernel(node):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.For):
                continue
            for inner in ast.walk(sub):
                if (
                    isinstance(inner, ast.AugAssign)
                    and isinstance(inner.op, ast.Add)
                    and _contains_multiply(inner.value)
                ):
                    findings.append(
                        _Finding(
                            issue_type="manual_k_reduction",
                            severity=5,
                            location=f"{node.name}: line {sub.lineno}",
                            description=(
                                "Hot loop accumulates a multiply (acc += a*b) "
                                "manually instead of using a tiled tl.dot, so "
                                "the kernel does not map to Cube."
                            ),
                            suggested_fix=(
                                "Rewrite the manual K-reduction as a tiled "
                                "tl.dot-based matmul so the kernel maps to Cube."
                            ),
                        )
                    )
                    break
    return findings


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _inside_jit_kernel(tree: ast.Module, target: ast.AST) -> bool:
    """Return True if ``target`` is lexically inside a @triton.jit function."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_jit_kernel(node):
            for inner in ast.walk(node):
                if inner is target:
                    return True
    return False


def _is_jit_kernel(func: ast.FunctionDef) -> bool:
    for decorator in func.decorator_list:
        dec_src = ast.unparse(decorator)
        if "triton.jit" in dec_src or dec_src.endswith("jit"):
            return True
    return False


def _contains_multiply(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Mult):
            return True
    return False


_DETECTOR_TYPE = Callable[[str, Optional[ast.Module]], List[_Finding]]

_DETECTORS: Tuple[_DETECTOR_TYPE, ...] = (
    _detect_permute_contiguous_materialization,
    _detect_implicit_transpose_in_dot,
    _detect_static_range_unroll,
    _detect_flat_1d_index_decode,
    _detect_invalid_num_warps,
    _detect_missing_autotune,
    _detect_missing_compile_hints,
    _detect_wrapper_loop_per_launch,
    _detect_manual_k_reduction,
)


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: scan_kernel_issues.py <kernel_path>", file=sys.stderr)
        sys.exit(2)
    findings = scan(sys.argv[1])
    if findings:
        print(json.dumps(findings, indent=2))
    else:
        print("[]")
