# Optimize Round Timing JSONL Design

## Summary

- Record optimize round timing as append-only JSONL files under `.triton-agent/round-timings/`.
- Use one file per round, named `opt-round-N.jsonl`.
- Stop storing round start and end timestamps in `.triton-agent/state.json`.
- Archive the full `round-timings/` directory into each optimize run archive instead of flattening timing data into one JSON file.

## Goals

- Make round timing easy to inspect without reading workflow state snapshots.
- Record round lifecycle timing and validation-command timing in one per-round event stream.
- Keep `.triton-agent/state.json` focused on workflow coordination state, not historical timing data.
- Preserve compatibility with legacy temporary workflow state that still contains `started_at` or `ended_at`.

## Non-Goals

- Do not redesign optimize phase names, round numbering, or submission rules.
- Do not add a new reporting command for timing logs in this change.
- Do not change baseline timing behavior or add timing logs for non-round commands.
- Do not fail `run-test-optimize` or `run-bench` solely because timing-log append failed.

## User-Visible Behavior

### Runtime timing directory

- When optimize workflow state is active, round timing files live under:
  - `.triton-agent/round-timings/opt-round-1.jsonl`
  - `.triton-agent/round-timings/opt-round-2.jsonl`
- Each line is one JSON object.
- Files are append-only and created on demand.

### Event sources

- `ascend-npu-optimize-state start-round` appends a `round_start` event.
- `ascend-npu-optimize-state submit-round` appends a `round_end` event.
- `ascend-npu-run-eval run-test-optimize` appends:
  - `run_test_start`
  - `run_test_end`
- `ascend-npu-run-eval run-bench` appends:
  - `run_bench_start`
  - `run_bench_end`

### Event shape

- Every event includes:
  - `event`
  - `timestamp`
  - `run_id`
  - `round`
- `run_test_*` events also include:
  - `command`
  - `test_file`
  - `operator_file`
- `run_bench_*` events also include:
  - `command`
  - `bench_file`
  - `operator_file`
- End events also include:
  - `return_code`

Example:

```json
{"event":"round_start","timestamp":"2026-07-06T12:34:56Z","run_id":"optimize-20260706-123456-abcdef","round":"opt-round-1"}
{"event":"run_test_start","timestamp":"2026-07-06T12:35:10Z","run_id":"optimize-20260706-123456-abcdef","round":"opt-round-1","command":"run-test-optimize","test_file":"differential_test_kernel.py","operator_file":"opt-round-1/opt_kernel.py"}
{"event":"run_test_end","timestamp":"2026-07-06T12:35:42Z","run_id":"optimize-20260706-123456-abcdef","round":"opt-round-1","command":"run-test-optimize","return_code":0,"test_file":"differential_test_kernel.py","operator_file":"opt-round-1/opt_kernel.py"}
```

## State Contract Changes

- New workflow-state writes for active or completed rounds should no longer include `started_at` or `ended_at`.
- Workflow validation should continue to accept legacy state payloads that already contain those fields so resumed or leftover temporary state is not rejected only because of this migration.
- Round activity remains derived from:
  - `phase`
  - `current_round`
  - `rounds[round].status`
  - `rounds[round].round_dir`

## Archival Behavior

- Optimize session archival should copy `.triton-agent/round-timings/` into:
  - `triton-agent-logs/<run_id>/round-timings/`
- No synthesized `round-timings.json` file should be generated anymore.
- If the runtime timing directory does not exist, archival should skip it without failing the session cleanup.

## Failure Handling

- `start-round` should treat timing-log append like the existing `attempts.md` mirror:
  - workflow state remains authoritative
  - append failures become warnings when possible
- `submit-round` should still complete the round even if timing-log append fails.
- `run-test-optimize` and `run-bench` should treat timing logging as best effort:
  - no active optimize round: skip logging
  - malformed or missing `.triton-agent/state.json`: skip logging
  - append failure: skip logging and keep command exit behavior unchanged

## Testing Strategy

- Add workflow-state regression coverage for:
  - `start-round` writing `round_start` JSONL
  - `submit-round` writing `round_end` JSONL
  - new state payloads omitting `started_at` and `ended_at`
  - legacy state payloads with `started_at` and `ended_at` still loading
- Add run-eval script coverage for:
  - `run-test-optimize` start/end timing events
  - `run-bench` start/end timing events
- Update optimize session archive coverage to assert directory archival at `round-timings/` instead of `round-timings.json`.
