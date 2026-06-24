from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TRITON_IMPORT_RE = re.compile(r"(?m)^\s*(?:import\s+triton\b|from\s+triton\b)")
_TRITON_DECORATOR_RE = re.compile(r"(?m)^\s*@\s*triton\.(?:jit|autotune|heuristics)\b")
_TRITON_LANGUAGE_RE = re.compile(r"(?m)\btriton\.language\b|\btl\.")
_TRITON_LAUNCH_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\[[^\]]+\]\s*\(", re.DOTALL)


@dataclass(frozen=True)
class KernelContinuityResult:
    ok: bool
    reason: str | None


def analyze_kernel_continuity(operator_path: Path) -> KernelContinuityResult:
    source = operator_path.read_text(encoding="utf-8")
    if not _TRITON_LAUNCH_RE.search(source):
        return KernelContinuityResult(
            ok=False,
            reason="round operator no longer preserves a recognizable Triton kernel launch path",
        )
    if _has_triton_signal(source):
        return KernelContinuityResult(ok=True, reason=None)
    return KernelContinuityResult(
        ok=False,
        reason="round operator no longer preserves a recognizable Triton kernel launch path",
    )


def _has_triton_signal(source: str) -> bool:
    return bool(
        _TRITON_IMPORT_RE.search(source)
        or _TRITON_DECORATOR_RE.search(source)
        or _TRITON_LANGUAGE_RE.search(source)
    )

