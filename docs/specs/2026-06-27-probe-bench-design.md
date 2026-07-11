# Probe-Bench Design

## Summary

- Add a new public CLI subcommand: `probe-bench`.
- Keep `run-bench` unchanged as the canonical benchmark execution surface.
- Make `probe-bench` a paired fast-screening command that always compares:
  - one candidate operator
  - against one required baseline operator
- Keep the CLI thin:
  - `src/` owns parser wiring, command dispatch, and skill-loader bridge code
  - skill-side helper code owns probe execution, artifact handling, caching, and classification
- Introduce one internal execution distinction:
  - `measurement_profile=canonical` for `run-bench`
  - `measurement_profile=probe` for `probe-bench`
- Cache only the baseline probe perf artifact under `.helix/`.
- Write the candidate probe perf artifact to `.helix/` too, but do not cache it.
- Support `--metric-source auto|kernel|total-op` on `probe-bench`.
- Return one screening classification:
  - `likely_gain`
  - `likely_regression`
  - `inconclusive`

## Goals

- Let optimize rounds reject obviously bad directions earlier without paying canonical benchmark cost every time.
- Preserve the current `run-bench` and `compare-perf` contracts as the authority for official performance conclusions.
- Reuse existing benchmark harnesses, benchmark cases, and shared perf parsing rules.
- Keep fast probe artifacts hidden from official optimize round artifacts and `round-state.json`.
- Make repeated baseline-side probe comparison cheap through a safe local cache.
- Keep default `probe-bench` output short and decision-oriented for agent workflows.

## Non-Goals

- Do not change the semantics of `run-bench`.
- Do not change the semantics of `compare-perf`.
- Do not replace canonical benchmark validation in `submit-round`.
- Do not write probe results into round-local canonical perf artifact paths such as `opt_<operator>_perf.txt`.
- Do not expose probe artifacts as official optimize evidence.
- Do not add a probe-specific optimization state machine.
- Do not support `--metric-source all` for `probe-bench`.
- Do not add candidate-side probe caching in the first version.
- Do not require users to learn or manually select a generic `measurement_profile` flag.

## Context

Optimize rounds now carry richer workflow state, but canonical benchmark cost remains unavoidable whenever the workflow needs an official speedup conclusion.

That creates a gap:

- some candidate edits are obviously bad and should be discarded quickly
- some candidate edits are obviously promising and should be kept long enough to finish the round
- but the workflow still needs canonical `run-bench` plus `compare-perf` before recording any official conclusion

The repository already has a clean split:

- `run-bench` executes benchmark workloads and writes perf artifacts
- `compare-perf` interprets perf artifacts and reports aggregate deltas and metric-source behavior

The new command should fit this split rather than overloading `run-bench`.

## User-Visible Semantics

`probe-bench` is a paired fast-screening command.

It always compares:

- one candidate operator file
- against one required baseline operator file
- using the same benchmark file
- under a fast probe measurement profile

It is not an official benchmark conclusion.

The command should help answer:

- is this direction clearly promising?
- is this direction clearly regressing?
- or is the fast signal too weak to trust?

The command should not answer:

- what is the official round speedup?
- what should be written into `round-state.json`?
- whether canonical `submit-round` benchmark work can be skipped

## Command Surface

### Public CLI

Add one new public subcommand:

```bash
helix probe-bench \
  --bench-file bench_op.py \
  --operator-file opt_op.py \
  --baseline-operator-file baseline_op.py
```

### Arguments

`probe-bench` should accept:

- `--bench-file`
- `--operator-file`
- `--baseline-operator-file` (required)
- `--bench-mode`
- `--metric-source`
- `--remote`
- `--remote-workdir`
- `--keep-remote-workdir`
- `--npu-devices`
- `--verbose`

`probe-bench` should not accept:

- `--output`
- any public `--measurement-profile` override

### Exit Codes

- return `0` when the command successfully completes and yields:
  - `likely_gain`
  - `likely_regression`
  - or `inconclusive`
- return `1` only for real failures such as:
  - benchmark execution failure
  - invalid cache metadata that cannot be recovered from
  - perf artifacts that cannot be compared
  - case mismatch or missing required metrics

`likely_regression` is a successful screening result, not a command failure.

## Architecture

### Thin `src/` Layer

Keep `src/` limited to public command wiring and bridge code.

Expected runtime ownership:

- `src/helix/models.py`
  - add `CommandKind.PROBE_BENCH`
- `src/helix/cli.py`
  - register the new parser and handler
- `src/helix/commands/execution.py`
  - add `handle_probe_bench(...)`
  - validate paths
  - call the execution bridge
  - render concise default output
- `src/helix/execution.py`
  - add `run_local_probe_bench(...)`
  - add `run_remote_probe_bench(...)`
  - load a skill-side helper through the existing bridge pattern

`probe-bench` should not stage skills.

It should mirror `run-bench` and `compare-perf` by loading a skill-side helper through `skill_loader.py`.

That means no new `STAGE_RULES` entry is expected in `src/helix/skill_staging.py`.

### Skill-Side Helper

Add a new helper under the run-eval skill:

```text
skills/common/ascend-npu-run-eval/scripts/probe_runner.py
```

This helper should own:

- baseline probe cache hit and miss logic
- baseline and candidate probe artifact path selection
- probe measurement-profile execution
- shared perf parsing and comparison reuse
- classification into:
  - `likely_gain`
  - `likely_regression`
  - `inconclusive`
- verbose-only diagnostic payload fields

### Existing Boundaries That Should Remain Unchanged

- `bench_runner.py` should stay responsible for benchmark execution behavior and perf artifact generation patterns.
- `skills/common/ascend-npu-run-eval/scripts/perf_artifacts.py` should stay the shared source for perf parsing and aggregate comparison semantics.
- `compare-perf` should remain the canonical performance comparison authority.

`probe-bench` should not become a second public aggregate-reporting command.

## Execution Split

The runtime must distinguish canonical benchmark execution from probe benchmark execution explicitly.

Introduce one internal execution distinction:

- `measurement_profile=canonical`
- `measurement_profile=probe`

This is an internal execution-layer concept only.

It should not become a user-facing generic flag on `run-bench`.

### Public Mapping

- `run-bench` always uses `measurement_profile=canonical`
- `probe-bench` always uses `measurement_profile=probe`

### Bench Cases Resolution

`probe-bench` should not add a public `--bench-cases-file` argument in the first version.

Instead, it should resolve the optional benchmark case sidecar the same way `bench_runner.py` already does:

```text
bench_cases_file = bench_file.with_suffix(".json")
```

Rules:

- if that sibling file exists, treat it as part of the benchmark input set
- if it does not exist, continue without a benchmark case sidecar
- when present, include it in remote staging and baseline-cache fingerprinting
- when absent, omit `bench_cases_file` and `bench_cases_fingerprint` from cache metadata

### Where The Difference Applies

The benchmark harness, benchmark module loading, and benchmark case set should stay the same between canonical and probe runs.

The difference should apply only after case resolution and before actual execution, by deriving an effective measurement configuration from the chosen profile.

### Measurement Profile Implementation

The clamp site for probe mode should be the resolved `BenchCase` records produced by:

```text
skills/common/ascend-npu-run-eval/scripts/bench_runtime.py
```

More specifically:

- `probe_runner.py` should resolve cases through the same `bench_runtime.load_bench_cases(...)` path used by canonical benchmark execution
- after resolution, `probe_runner.py` should create a probe-local copy of those `BenchCase` records
- the probe-local records should then clamp `warmup` and `repeats` before execution
- the existing timing and profiler paths should consume those effective values without any new public benchmark contract

This gives `probe-bench` a real fast path in v1 without changing canonical `run-bench` semantics.

### Probe Measurement Profile

The probe profile should be implemented by shrinking case-level measurement cost instead of inventing a second benchmark harness contract.

Recommended first-version behavior:

- clamp warmup with a probe cap
- clamp repeats with a probe cap

For example:

```text
effective_warmup = min(case_warmup, probe_warmup_cap)
effective_repeats = min(case_repeats, probe_repeats_cap)
```

This keeps probe behavior close to the same case contract while still reducing cost materially.

## Probe Artifacts

Probe artifacts must not reuse or overwrite canonical perf artifact paths.

All probe artifacts should live under the workspace-local hidden directory:

```text
.helix/
```

### Baseline-Side Files

Use fixed paths:

- `.helix/baseline_probe_perf.txt`
- `.helix/baseline_probe_perf.meta.json`
- `.helix/baseline_probe_perf.lock`

`baseline_probe_perf.txt` is the cached baseline fast-probe perf artifact.

The sidecar metadata file is the only cache-hit authority.

The lock file is an internal coordination detail for safe shared-cache refreshes.

### Candidate-Side Files

Do not use one global fixed candidate path.

Use a per-invocation hidden path such as:

- `.helix/candidate_probe_perf.<pid>.<run_token>.txt`

This avoids clobbering concurrent `probe-bench` invocations in the same workspace.

Do not add candidate-side caching in the first version.

### Remote Copy-Back

For `--remote`, probe execution may happen in a remote workspace, but the authoritative probe artifacts for caching and comparison should still be the local hidden files under `.helix/`.

`probe-bench` should reuse the same remote archive and copy-back pattern that `bench_runner.run_remote_bench(...)` already uses:

- generate probe perf artifacts remotely
- copy them back into the local hidden probe-artifact paths
- run cache validation and comparison against those local copies

### Concurrency And Atomicity

The shared baseline cache should support concurrent `probe-bench` invocations in one workspace.

Required behavior:

- baseline cache refresh should be serialized with the baseline lock file
- perf and sidecar updates should use write-then-atomic-rename behavior
- once an invocation has a cache hit or completes a cache rebuild, it should snapshot the baseline probe data it will compare against before releasing the lock
- later cache rewrites by other invocations must not change an in-flight comparison

### Why Fixed Hidden Files

Fixed hidden files are preferable to canonical paths or temp-only files because they:

- avoid polluting official optimize artifacts
- keep verbose diagnostics easy to understand
- make baseline cache reuse straightforward
- remain clearly separate from round contract outputs

## Baseline Cache Contract

Only the baseline-side probe artifact should be cached.

The cache should be reused only when a metadata sidecar matches exactly.

### Sidecar Shape

Recommended metadata shape:

```json
{
  "schema_version": 1,
  "measurement_profile": "probe",
  "probe_contract": {
    "name": "fast-probe",
    "warmup_cap": 1,
    "repeats_cap": 3
  },
  "baseline_operator_file": "/abs/path/baseline_op.py",
  "baseline_operator_fingerprint": "sha256:...",
  "bench_file": "/abs/path/bench_op.py",
  "bench_file_fingerprint": "sha256:...",
  "bench_mode": "torch-npu-profiler",
  "remote": null,
  "remote_workdir": null,
  "npu_devices": "0",
  "generated_at": "2026-06-27T10:00:00Z"
}
```

When the derived benchmark case sidecar exists, also include:

- `bench_cases_file`
- `bench_cases_fingerprint`

### Cache Invalidation Rules

The baseline probe cache must be invalidated and rebuilt when any of these change:

- baseline operator contents
- benchmark harness contents
- benchmark case sidecar contents
- probe contract
- bench mode
- local vs remote execution target
- remote workspace root
- device selection
- metadata schema version

If either:

- the perf file is missing
- the sidecar is missing
- the sidecar is malformed

then treat the cache as a miss and regenerate it.

### Fingerprints

Cache validity should depend on content fingerprints, not only file paths or mtimes.

Paths are useful for diagnostics but should not be trusted as the sole cache key.

## Metric Source Semantics

`probe-bench` should support:

- `--metric-source auto`
- `--metric-source kernel`
- `--metric-source total-op`

Default:

- `auto`

Do not support:

- `--metric-source all`

`probe-bench` must yield one classification result, so dual-section public output is the wrong shape here.

### Behavior

`probe-bench` should reuse the same metric-source meaning as `compare-perf`:

- `kernel`
  - require kernel timing
- `total-op`
  - require total-op timing
- `auto`
  - prefer kernel timing
  - fall back to total-op timing when needed

### Cache Interaction

`metric_source` should not become part of the baseline cache key.

The cache stores a probe perf artifact, not a compare result.

Changing `metric_source` changes how probe artifacts are interpreted, not whether the baseline probe artifact itself is reusable.

### Output

Default output should include a short metric-source line, such as:

- `Metric source: kernel`
- `Metric source: total-op`
- `Metric source: mixed`

`mixed` should only appear when `auto` resolves different cases through different sources.

When `mixed` occurs, the command may still succeed and classify, but should also emit a warning.

## Classification Rules

`probe-bench` should be a conservative screening tool.

It should only emit a strong conclusion when the probe signal is clearly positive or clearly negative.

### Classification Set

- `likely_gain`
- `likely_regression`
- `inconclusive`

### Per-Case Direction Threshold

For each comparable case, determine its direction using the same resolved metric source that produced the aggregate comparison:

- improvement greater than `1%` counts as `improved`
- regression greater than `1%` counts as `regressed`
- values within `[-1%, +1%]` count as `unchanged`

`improved_cases` and `regressed_cases` should count only those thresholded comparable cases.

### Recommended Default Thresholds

`likely_gain`

- `geomean_speedup >= 1.10`
- `improved_cases > regressed_cases`
- no single case regresses by more than `8%`

`likely_regression`

- `geomean_speedup <= 0.95`
- or any single case regresses by at least `15%`

`inconclusive`

- every other comparable result

This includes:

- small positive movement
- small negative movement
- case-level disagreement
- mixed-source but comparable output
- grey-zone results that are plausible fast-probe noise

### Why Conservative Thresholds

- fast probe should not bless weak gains too easily
- fast probe should reject obviously bad directions early
- the broad middle should remain undecided until stronger evidence exists

### Non-Comparable Results

These should fail the command rather than produce a classification:

- baseline probe execution failure
- candidate probe execution failure
- non-comparable perf artifacts
- mismatched case ids
- missing required metric source values

The distinction should stay explicit:

- negative screening result = successful command
- inability to compare = failed command

## Default Output

Default output should stay short and decision-oriented.

### Output Contract

`Probe classification:` is the only first-class direction field in the public output contract.

Everything else is secondary:

- `Metric source:` is supporting diagnostic context
- aggregate speedup and case-split numbers are advisory and non-authoritative
- `Summary:` is a human-readable reminder that canonical `run-bench` remains required

Agents should use `Probe classification:` as the direction signal and should not treat the advisory aggregates as official performance evidence.

### Recommended Lines

Recommended lines:

- `Probe classification: likely_gain`
- `Metric source: kernel`
- `Advisory geomean speedup: 1.14x`
- `Advisory avg improvement: +12.4%`
- `Advisory improved cases: 5`
- `Advisory regressed cases: 0`
- `Summary: Fast probe indicates a likely gain over the baseline. Use canonical run-bench before recording any official speedup.`

Optional:

- `Warnings: ...`

Default output should not include:

- cache hit and miss details
- baseline probe path
- candidate probe path
- metadata mismatch diagnostics

### Verbose Output

`--verbose` may additionally show:

- baseline cache hit or miss
- baseline probe perf path
- candidate probe perf path
- metadata mismatch reason
- remote workspace details
- lower-level execution diagnostics

## Relationship To Optimize Workflow

`probe-bench` is an execution helper, not an official optimize contract step.

It should:

- help an agent decide whether a direction is worth continuing
- help an agent discard a clearly bad direction faster

It should not:

- replace canonical `run-bench`
- write official perf conclusions into round state
- replace `submit-round`
- create a second benchmark authority beside `compare-perf`

Canonical optimize conclusions remain:

- canonical `run-bench`
- followed by canonical interpretation through existing comparison semantics

## Testing

Add focused tests for:

- CLI parser coverage for `probe-bench` in `tests/test_cli.py`
- handler coverage in `tests/test_execution_commands.py`
- remote probe copy-back and bridge coverage in `tests/test_remote_execution.py`
- probe helper behavior in a dedicated module such as `tests/test_probe_runner.py`

Test scenarios should include:

- required `--baseline-operator-file`
- default `metric_source=auto`
- acceptance of `metric_source=kernel`
- acceptance of `metric_source=total-op`
- rejection of `metric_source=all`
- benchmark case sidecar derivation from `bench_file.with_suffix(".json")`
- baseline cache hit
- baseline cache miss from changed baseline operator
- baseline cache miss from changed benchmark harness
- baseline cache miss from changed benchmark case sidecar
- baseline cache miss from changed probe contract
- baseline remote probe copy-back into the local hidden cache path
- candidate per-invocation path isolation under concurrent runs
- default output excludes hidden-file details
- verbose output includes hidden-file details
- `likely_gain`
- `likely_regression`
- `inconclusive`
- command failure on non-comparable perf artifacts
- local and remote execution path coverage

Because the new helper lives under `skills/*/scripts/`, completion should also include the required file-scoped strict Pyright check for that helper.

## Files Expected To Change

- `docs/specs/2026-06-27-probe-bench-design.md`
- `src/helix/models.py`
- `src/helix/cli.py`
- `src/helix/commands/execution.py`
- `src/helix/execution.py`
- `src/helix/run_eval_mcp_server.py` if MCP exposure is later desired
- `skills/common/ascend-npu-run-eval/scripts/probe_runner.py`
- tests covering CLI, execution bridge, and probe helper behavior

## Out Of Scope

This change does not:

- alter canonical `run-bench` behavior
- alter canonical `compare-perf` behavior
- add candidate-side probe caching
- add public probe threshold tuning flags
- add `--metric-source all` support to `probe-bench`
- treat probe artifacts as official optimize round artifacts
- replace canonical benchmark validation in optimize workflow
