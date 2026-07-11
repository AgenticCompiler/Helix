---
title: Baseline Contract
created: 2026-04-13
summary: Outline the minimal layer that claims a canonical baseline contract before optimize rounds run.
---

# Baseline Contract Design

## Objective

Capture the canonical baseline artifact semantics described in `docs/specs/2026-04-13-optimize-baseline-prep-design.md` via a focused helper layer. That layer should cover the workspace layout (`baseline/` directory), strict `baseline/state.json` parsing, and discovery of the required perf and operator artifacts.

## Data Modeling

- **`BaselineState` dataclass**: mirrors the required `state.json` keys (`baseline_kind`, `source_operator`, `baseline_operator`, `test_file`, `test_mode`, `bench_file`, `bench_mode`, `perf_artifact`, `correctness_status`, `benchmark_status`, `baseline_established`) plus optional notes fields. This struct will live in `src/helix/optimize/models.py` so the rest of the optimizer can rehydrate it.
- **`BaselineArtifactsInspection` dataclass**: holds the resolved paths (workspace baseline dir, `state.json`, canonical perf artifact, operator snapshot) plus the parsed `BaselineState`.

## Validation Helpers

`src/helix/optimize/baseline.py` will contain helpers such as:

- `baseline_dir(workspace: Path) -> Path`
- `load_baseline_state(workspace: Path) -> BaselineState`: opens `baseline/state.json`, enforces object schema, required fields, and that `baseline_established` is truthy.
- `_find_baseline_operator(baseline_dir: Path) -> Path`: picks the single non-metadata file as the operator snapshot and raises when missing/multiple exist.
- `_resolve_perf_path(workspace: Path, state: BaselineState) -> Path`: resolves the `perf_artifact` field (defaulting to `baseline/perf.txt`), ensures the file is inside the workspace, and validates existence.
- `inspect_baseline_artifacts(workspace: Path) -> BaselineArtifactsInspection`: packages the dir, file paths, and state, raising if anything is missing.

## Testing

Create `tests/test_optimize_baseline.py` with focused unit tests that:

1. Confirm `load_baseline_state` rejects payloads missing required keys and that the helper raises when the workspace lacks the referenced perf/operator files.
2. Confirm `inspect_baseline_artifacts` raises when the canonical perf or operator is missing.
3. Confirm `inspect_baseline_artifacts` returns the expected `perf_path` when `baseline/perf.txt` is present, and that the operator snapshot resolution prefers the single eligible file under `baseline/`.

These tests will fail until the module exists, satisfying our TDD flow for Task 1.
