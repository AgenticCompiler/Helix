"""Stable Helix-facing API for optimize-state contracts and workflow state."""

from __future__ import annotations

from baseline.check import (
    baseline_gate_issues,
    check_baseline,
    inspect_baseline_artifacts,
    load_baseline_state,
)
from round.check import (
    best_completed_round_geomean_speedup,
    check_round,
    cleanup_dir_pt_files,
    cleanup_pt_file,
    cleanup_workspace_profile_artifacts,
    count_completed_round_directories,
    count_terminal_round_directories,
    inspect_round_artifacts,
    iter_terminal_round_directories,
    load_round_state,
    ordinary_optimize_pt_cleanup_mode,
    resolve_round_operator_file,
    resolve_round_perf_file,
)
from state_manage.state_machine import (
    bootstrap_state,
    load_state,
    mark_baseline_passed,
    render_phase_summary,
)


__all__ = (
    "baseline_gate_issues",
    "best_completed_round_geomean_speedup",
    "bootstrap_state",
    "check_baseline",
    "check_round",
    "cleanup_dir_pt_files",
    "cleanup_pt_file",
    "cleanup_workspace_profile_artifacts",
    "count_completed_round_directories",
    "count_terminal_round_directories",
    "inspect_baseline_artifacts",
    "inspect_round_artifacts",
    "iter_terminal_round_directories",
    "load_baseline_state",
    "load_round_state",
    "load_state",
    "mark_baseline_passed",
    "ordinary_optimize_pt_cleanup_mode",
    "render_phase_summary",
    "resolve_round_operator_file",
    "resolve_round_perf_file",
)
