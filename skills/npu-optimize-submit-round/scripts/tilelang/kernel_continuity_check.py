from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# TileLang kernel signal patterns (from TileLang-Ascend Programming Guide).
_TILELANG_IMPORT_RE = re.compile(r"(?m)^\s*(?:import\s+tilelang\b|from\s+tilelang\b)")
_TILELANG_DECORATOR_RE = re.compile(r"(?m)^\s*@\s*(?:T\.prim_func|jit)\b")
_TILELANG_LANGUAGE_RE = re.compile(r"(?m)\btilelang\.language\b|\bT\.\b")
_TILELANG_LAUNCH_RE = re.compile(r"(?m)with\s+T\.Kernel\s*\([^)]*\)\s*as\s+\(")


@dataclass(frozen=True)
class KernelContinuityResult:
    ok: bool
    reason: str | None


def analyze_kernel_continuity(operator_path: Path) -> KernelContinuityResult:
    source = operator_path.read_text(encoding="utf-8")
    if not _TILELANG_LAUNCH_RE.search(source):
        return KernelContinuityResult(
            ok=False,
            reason="round operator no longer preserves a recognizable TileLang kernel launch path",
        )

    if _has_tilelang_signal(source):
        return KernelContinuityResult(ok=True, reason=None)

    return KernelContinuityResult(
        ok=False,
        reason="round operator no longer preserves a recognizable TileLang kernel launch path",
    )


def _has_tilelang_signal(source: str) -> bool:
    return bool(
        _TILELANG_IMPORT_RE.search(source)
        or _TILELANG_DECORATOR_RE.search(source)
        or _TILELANG_LANGUAGE_RE.search(source)
    )

