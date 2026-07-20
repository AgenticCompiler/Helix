"""Stable Helix-facing API for benchmark execution."""

from __future__ import annotations

from bench_contract import parse_bench_metadata, resolve_bench_kernel_names
from run_bench_modes import normalize_bench_mode
from run_bench_local_api import run_local_bench
from run_bench_remote_api import run_remote_bench


__all__ = (
    "parse_bench_metadata",
    "resolve_bench_kernel_names",
    "normalize_bench_mode",
    "run_local_bench",
    "run_remote_bench",
)
