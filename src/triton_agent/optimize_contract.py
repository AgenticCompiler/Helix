from __future__ import annotations

BASELINE_STATE_REQUIRED_FIELDS = (
    "baseline_kind",
    "source_operator",
    "baseline_operator",
    "test_file",
    "test_mode",
    "bench_file",
    "bench_mode",
    "perf_artifact",
    "correctness_status",
    "benchmark_status",
    "baseline_established",
)

_BASELINE_STATE_FIELD_DESCRIPTIONS = (
    (
        "baseline_kind",
        "record whether the canonical baseline is the original operator or a minimally repaired prepared baseline.",
    ),
    (
        "source_operator",
        "record the workspace-relative path to the operator file that baseline preparation started from.",
    ),
    (
        "baseline_operator",
        "record the workspace-relative path to the operator snapshot saved under `baseline/`.",
    ),
    (
        "test_file",
        "record the workspace-relative path to the correctness harness used for the baseline.",
    ),
    (
        "test_mode",
        "record the resolved correctness mode used for the baseline run.",
    ),
    (
        "bench_file",
        "record the workspace-relative path to the benchmark harness used for the baseline.",
    ),
    (
        "bench_mode",
        "record the resolved benchmark mode used for the baseline run.",
    ),
    (
        "perf_artifact",
        "record the canonical baseline perf artifact path, normally `baseline/perf.txt`.",
    ),
    (
        "correctness_status",
        "record the final baseline correctness result; use `passed` only after correctness succeeds.",
    ),
    (
        "benchmark_status",
        "record the final baseline benchmark result; use `passed` only after the benchmark succeeds.",
    ),
    (
        "baseline_established",
        "set this to `true` only after `correctness_status` is `passed`, `benchmark_status` is `passed`, and the canonical baseline artifacts are written.",
    ),
)


def baseline_state_contract_lines() -> tuple[str, ...]:
    lines = ["Write `baseline/state.json` with these required fields:"]
    lines.extend(
        f"`{field_name}`: {description}"
        for field_name, description in _BASELINE_STATE_FIELD_DESCRIPTIONS
    )
    lines.append(
        "Set `baseline_established` to `true` only after `correctness_status` is `passed` and `benchmark_status` is `passed`."
    )
    return tuple(lines)
