# Run-Bench JSONL Perf Artifact Design

## Summary

Replace the current ad hoc line-plus-comment perf artifact content with structured JSONL records while keeping the existing perf filenames unchanged. Each JSONL line represents one benchmark case summary, preserves case-level per-op timing rows plus derived aggregate timings, lets `compare-perf` and other consumers read the new format through a shared compatibility layer, and keeps legacy text perf files readable.

## Goals

- Make `run-bench` perf artifacts structured enough that producers and consumers no longer depend on comment-line conventions such as `# raw-op-statistic-*` and `# latency-error-*`.
- Keep perf artifacts append-friendly and case-oriented by storing one JSON object per benchmark case.
- Preserve the case-level op timing rows that users inspect today from `raw-op-statistic-*` comments.
- Preserve the current benchmark artifact filenames such as `baseline/perf.txt`, `<operator>_perf.txt`, and round-local `perf.txt`.
- Preserve current comparison semantics for:
  - kernel latency
  - total-op fallback
  - skip-latency-error handling
- Record one explicit wall-clock timing field per case with a name that describes its meaning unambiguously.
- Keep historical text perf artifacts readable without migration.

## Non-Goals

- Do not rename public perf artifact paths from `*.txt` to `*.jsonl` in this change.
- Do not migrate historical perf artifacts in-place.
- Do not add schema-version metadata to perf artifact contents.
- Do not change case-id matching rules or the public `compare-perf` CLI surface.
- Do not embed full raw profiler output trees or CSV files inside the new perf artifact format.

## Current Problem

Today perf artifacts are a custom text format built from:

- `latency-<id>: <value>` lines
- `# raw-op-statistic-<id>: <json>` comments
- `# latency-error-<id>: <message>` comments
- `# elapsed-seconds-<id>: <float>` comments
- kernel-name and kernel-source comments

This has three problems:

1. The format is difficult to evolve because meaning is spread across unrelated comment prefixes.
2. Comparison code has to reconstruct one logical case record by correlating multiple line types.
3. Human readers still do not get especially nice output, even though the format sacrifices structure in the name of text readability.

## Decision

- Keep the existing perf filenames.
- Change newly generated perf artifact contents to JSONL.
- Write one JSON object per benchmark case in stable case order.
- Preserve the per-op timing rows that the old `raw-op-statistic-*` comments exposed.
- Persist the already-computed aggregate values that downstream consumers actually use.
- Keep a shared parser that accepts both:
  - new JSONL perf artifacts
  - legacy text perf artifacts

## JSONL Record Schema

Each JSONL line represents one case record with this shape:

```json
{
  "case_label": "case-a",
  "kernel_names": ["KernelA", "KernelB"],
  "kernel_source": "metadata",
  "kernel_avg_time_us": 12.5,
  "ops": [
    {"op_type": "KernelA", "avg_time_us": 5.0},
    {"op_type": "KernelB", "avg_time_us": 6.0},
    {"op_type": "HelperOp", "avg_time_us": 8.75}
  ],
  "total_op_avg_time_us": 19.75,
  "error_message": null,
  "case_wall_clock_seconds": 0.482193
}
```

Field rules:

- `case_label`
  - string
  - unique within one perf artifact
  - continues to define the derived comparison id `latency-<case_label>`
- `kernel_names`
  - array of strings
  - preserves the resolved kernel names used for kernel-latency matching
- `kernel_source`
  - string
  - preserves whether kernel names came from benchmark metadata, runtime discovery, or another existing source
- `kernel_avg_time_us`
  - float or `null`
  - `null` means the case did not produce a comparable kernel latency
- `ops`
  - array of objects or `null`
  - each object has:
    - `op_type`: string
    - `avg_time_us`: float
  - preserves the normalized case-level op timing rows that users inspect for per-op breakdowns
- `total_op_avg_time_us`
  - float or `null`
  - stores the already-aggregated whole-case operator time that `compare-perf --metric-source total-op` needs
  - when `ops` is present, this value must equal `sum(op["avg_time_us"] for op in ops)`
- `error_message`
  - string or `null`
  - stores the case-level failure or non-comparable explanation
  - kernel-miss cases should keep an explicit explanation such as `no resolved kernels matched ...` even when total-op fallback remains available
- `case_wall_clock_seconds`
  - float or `null`
  - the wall-clock execution time for that one case attempt

## `case_wall_clock_seconds` Semantics

`case_wall_clock_seconds` replaces the older `elapsed_seconds` name because the new name states both:

- this timing is per case
- this timing is measured as wall-clock runtime rather than kernel or profiler time

Capture rules:

- For `msprof`, measure from immediately before launching the per-case benchmark command until that command returns.
- For `standalone`, measure around the per-case standalone profiling execution path.
- Record the value for successful and failed cases whenever execution of the case attempt actually started.
- Treat the field as informational metadata only. It does not participate in `compare-perf`, optimize status, or benchmark pass/fail decisions.

## Why Keep Both `ops` And `total_op_avg_time_us`

The new perf artifact needs both:

- case-level op detail for users and downstream analysis
- one direct aggregate value for comparison code

`ops` stays because users still need to inspect the per-op timing rows for one case without reopening profiler CSV artifacts.

`total_op_avg_time_us` stays because:

- `compare-perf --metric-source total-op` can consume one explicit number directly
- status and summary code do not need to re-sum `ops` every time
- the artifact makes the intended total-op comparison basis explicit

This is intentional redundancy. Writers should emit both fields consistently instead of forcing every consumer to recompute one from the other.

## Comparison Semantics Under JSONL

`compare-perf` should preserve the current metric-source behavior using JSONL fields:

- `--metric-source kernel`
  - require `kernel_avg_time_us`
- `--metric-source total-op`
  - require `total_op_avg_time_us`
- `--metric-source auto`
  - prefer `kernel_avg_time_us`
  - fall back to `total_op_avg_time_us` when kernel latency is unavailable but total-op timing exists
- `--metric-source all`
  - print both sections using the same stored case records

Error handling should preserve current behavior:

- A case with a non-null `error_message` and no usable metric should remain a comparison error.
- A kernel-miss case may still compare under `auto` or `total-op` when:
  - `kernel_avg_time_us` is `null`
  - `ops` preserves the per-op timing rows for the case
  - `total_op_avg_time_us` is non-null
  - `error_message` explains the kernel miss
- `--skip-latency-errors` should continue skipping invalid cases and return failure after printing the skipped-case summary.

## Compatibility Strategy

Shared perf parsing in `skills/triton-npu-run-eval/scripts/perf_artifacts.py` should become format-aware:

1. Read the first non-empty line.
2. If it begins with `{`, parse the file as JSONL case records.
3. Otherwise, parse the file as the legacy text format.

The public parser helpers should continue exposing the same normalized internal comparison view regardless of source format.

Compatibility rules:

- Legacy text perf files remain valid inputs to `compare-perf`, `status`, `verify`, and any optimize workflow that reads archived perf artifacts.
- Newly generated perf files use JSONL by default.
- Legacy `# raw-op-statistic-<id>: {"ops":[...]}` comments should map to JSONL `ops`, and legacy total-op fallback should continue deriving from those rows.
- Additive future JSON fields should be tolerated by parsers and ignored unless a consumer explicitly needs them.

## Internal Normalization

To keep the rest of the code simple, the parser layer should normalize both formats into the same case-oriented internal record before building comparison entries.

That normalized record should contain, at minimum:

- case label
- kernel latency value or missing marker
- op timing rows when available
- total-op value or missing marker
- error message
- kernel metadata
- case wall-clock timing

`compare-perf`, status computation, verify flows, and round artifact inspection should continue depending on normalized values rather than branching on source file format.

## Producer Changes

Update perf writers so newly generated artifacts emit JSONL records instead of mixed text/comment blocks.

Implementation scope:

- keep `perf_output_path()` and public filenames unchanged
- replace line-oriented rendering helpers with JSONL record rendering helpers
- write `ops` plus `total_op_avg_time_us` on every case that has op timing data
- keep kernel-miss explanations in `error_message` instead of silently dropping them when total-op fallback exists
- rename `PerfCaseRecord.elapsed_seconds` to `PerfCaseRecord.case_wall_clock_seconds`
- update both `msprof` and `standalone` producers to populate the renamed field

## Consumer Changes

Update shared perf consumers to read either format through the compatibility layer.

Implementation scope:

- `compare-perf`
- status and optimize-status perf readers
- verify flows that compare baseline and optimized perf artifacts
- any helper that currently expects raw `latency-*` text lines

The goal is that higher-level code should not care whether a given artifact came from:

- historical text output
- newly generated JSONL output

## Verification

- Add parser tests for JSONL perf files with:
  - successful kernel-latency cases
  - cases that preserve `ops` rows
  - total-op-only fallback cases
  - failed cases with error messages
- Add tests proving kernel-miss JSONL output preserves:
  - `ops`
  - `total_op_avg_time_us`
  - an explanatory `error_message`
- Keep existing legacy text parser tests and ensure they still pass.
- Add tests proving `compare-perf` produces the same metric-source results for equivalent legacy-text and JSONL fixtures.
- Add tests proving `run-bench` now writes JSONL records in stable case order.
- Re-run status and verify tests that consume perf artifacts to confirm new-format compatibility.
