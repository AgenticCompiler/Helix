# triton-agent

`triton-agent` is a CLI for generating, running, and optimizing Triton Ascend NPU operator workflows with code agents and local skills.

This README is organized by task so you can quickly find the right command for the job.

## Command Map

- `gen-test`: generate a correctness test for one operator.
- `run-test`: run an existing generated test.
- `gen-eval`: generate both test and benchmark assets for one operator.
- `gen-eval-batch`: generate evaluation assets for many operator workspaces.
- `gen-bench`: generate a benchmark for one operator.
- `run-bench`: run an existing generated benchmark.
- `optimize`: optimize one operator.
- `optimize-status`: summarize optimization progress across many workspaces.
- `verify`: rerun tests and benchmarks for the current best optimize round.
- `verify-batch`: verify many optimize workspaces under one root.
- `optimize-batch`: optimize many operator workspaces.
- `compare-result`: compare two archived correctness result files.
- `compare-perf`: compare two archived performance files.

## Quick Start

Most workflows start from a single operator file:

```bash
uv run triton-agent gen-test --input a.py
uv run triton-agent run-test --test-file test_a.py --operator-file a.py

uv run triton-agent gen-bench --input a.py
uv run triton-agent run-bench --bench-file bench_a.py --operator-file a.py

uv run triton-agent optimize --input a.py
```

For batch workflows, point `--input` at either a directory whose immediate child directories are operator workspaces, or a single operator workspace directory:

```bash
uv run triton-agent gen-eval-batch --input operators_root
uv run triton-agent gen-eval-batch --input .
uv run triton-agent optimize-status --input operators_root
uv run triton-agent optimize-status --input operators_root --format markdown
uv run triton-agent verify --input .
uv run triton-agent verify-batch --input operators_root
uv run triton-agent optimize-batch --input operators_root
```

## Generate Tests

Use `gen-test` when you need a correctness harness for one operator.

```bash
uv run triton-agent gen-test --input a.py
```

Common options:

- `--output test_a.py`: write to a specific path.
- `--test-mode standalone|differential`: choose the generated test style. Default is `standalone`.
- `--agent codex|opencode|pi|claude|openhands|traecli`: choose the backend.
- `--interact`: open an interactive agent session.
- `--show-output`: stream non-interactive agent output.
- `--force-overwrite`: replace an existing generated file.
- `--remote user@host[:port]`: generate with remote execution context in mind.
- `--remote-workdir <path>`: set the remote working root.

Example:

```bash
uv run triton-agent gen-test --input a.py --test-mode differential --agent codex
```

## Run Tests

Use `run-test` when you already have a generated test file and want to execute it.

```bash
uv run triton-agent run-test --test-file test_a.py --operator-file a.py
```

Common options:

- `--test-mode standalone|differential`: override the mode recorded in the test file.
- `--remote user@host[:port]`: run through SSH on a remote machine.
- `--remote-workdir <path>`: set the remote working root.
- `--keep-remote-workdir`: keep the remote workspace for debugging.
- `--verbose`: print more execution detail.

Example:

```bash
uv run triton-agent run-test --test-file differential_test_a.py --operator-file opt_a.py
```

## Generate Evaluation Assets

Use `gen-eval` when you want both correctness and benchmark assets in one step.

```bash
uv run triton-agent gen-eval --input a.py
```

What it is for:

- preparing a full evaluation setup for one operator
- generating both test and benchmark harnesses together

Common options:

- `--agent codex|opencode|pi|claude|openhands|traecli`
- `--test-mode standalone|differential`: default is `differential`
- `--bench-mode standalone|msprof`: default is `standalone`
- `--interact`
- `--show-output`
- `--force-overwrite`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

Example:

```bash
uv run triton-agent gen-eval --input a.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
```

## Generate Benchmarks

Use `gen-bench` when you only need a benchmark harness.

```bash
uv run triton-agent gen-bench --input a.py
```

Common options:

- `--output bench_a.py`
- `--bench-mode standalone|msprof`: default is `standalone`
- `--agent codex|opencode|pi|claude|openhands|traecli`
- `--interact`
- `--show-output`
- `--force-overwrite`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

Example:

```bash
uv run triton-agent gen-bench --input a.py --bench-mode standalone
```

## Run Benchmarks

Use `run-bench` when you already have a generated benchmark file and want to execute it.

```bash
uv run triton-agent run-bench --bench-file bench_a.py --operator-file a.py
```

Common options:

- `--bench-mode standalone|msprof`: override the mode recorded in the benchmark file.
- `--remote user@host[:port]`
- `--remote-workdir <path>`
- `--keep-remote-workdir`
- `--verbose`

Example:

```bash
uv run triton-agent run-bench --bench-file bench_a.py --operator-file opt_a.py
```

## Optimize One Operator

Use `optimize` when you want the agent to iterate on one operator and produce optimization rounds.

```bash
uv run triton-agent optimize --input a.py
```

You may also point `--input` at a single operator workspace directory when that directory contains exactly one candidate operator file, for example `uv run triton-agent optimize --input .`.

Common options:

- `--output opt_a.py`: write the optimized operator to a specific path.
- `--agent codex|opencode|pi|claude|openhands|traecli`
- `--prompt "..."`: append extra worker instructions without replacing the built-in optimize contract.
- `--test-mode standalone|differential`: default is `differential`
- `--bench-mode standalone|msprof`: default is `standalone`
- `--resume auto|continue|fresh`: default is `auto`
- `--reset-optimize`: only valid with `--resume fresh`; remove known optimize-session artifacts before starting a new run while keeping reusable test and benchmark harnesses.
- `--require-analysis`: strengthen analysis-first optimize guidance before the first code-changing round.
- `--min-rounds <N>`: require at least N optimization rounds.
- `--no-agent-session`: disable persistent agent sessions when supported.
- `--interact`
- `--show-output`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

Examples:

```bash
uv run triton-agent optimize --input a.py --min-rounds 3
uv run triton-agent optimize --input a.py --resume continue
uv run triton-agent optimize --input a.py --require-analysis
uv run triton-agent optimize --input a.py --prompt "Prioritize memory-coalescing improvements."
```

Resume modes:

- `auto`: continue only when there is a complete existing optimize session; otherwise start fresh or fail if the workspace is incomplete.
- `continue`: require an existing resumable optimize session.
- `fresh`: require a clean workspace with no existing optimize artifacts.
- `fresh` + `--reset-optimize`: delete known optimize-session artifacts first, but keep reusable generated test and benchmark harnesses.

Optimize behavior:

- Establish or reuse a canonical `baseline/` directory before treating `opt-round-1` as the first optimization round.
- Keep canonical baseline assets under:
  - `baseline/state.json`
  - `baseline/perf.txt`
  - one baseline operator snapshot under `baseline/`
- Reuse existing test and benchmark harnesses when they already exist in the workspace.
- Generate missing harnesses only when the required validation artifact is absent.
- Allow the agent to do minimal repair work during baseline preparation when that is required to reach a correct, benchmarkable starting point.
- Keep canonical optimize-session performance comparisons anchored to `baseline/perf.txt`, even when a round also compares locally against its chosen parent.
- Record each optimize code agent launch under `optimize-logs/triton-agent/<run-id>/agent-sessions.jsonl` with timestamp, role, session id, and agent backend. Missing session ids are recorded as `unknown`.
- Run optimize as explicit worker rounds with a supervisor audit between rounds instead of relying on one unconstrained agent pass.
- Keep the shared workspace guidance role-neutral; worker versus supervisor role assignment comes from the launch prompt plus the live `.triton-agent/round-brief.md` and `.triton-agent/supervisor-report.md` handoff files.
- Use fresh agent invocations for worker and supervisor passes so role-specific optimize context does not leak across the session.
- Treat each round as a hypothesis-driven experiment: explain why the change may help and what evidence supports it.
- Require each completed round to leave auditable artifacts such as `attempts.md`, `summary.md`, comparable perf data, and structured round state.
- Allow the supervisor to repair metadata derived from existing facts, but never to invent missing benchmark, profiler, IR, or correctness evidence.
- If profiling or IR capture is skipped for a round, explain why the existing evidence is already sufficient.

## Work On Many Operators

Use the batch commands when `--input` points to a directory of operator workspaces. `gen-eval-batch` and `optimize-batch` can also accept one operator workspace directory directly.

### Generate Evaluation Assets In Batch

```bash
uv run triton-agent gen-eval-batch --input operators_root
```

Common options:

- `--agent codex|opencode|pi|claude|openhands|traecli`
- `--test-mode standalone|differential`
- `--bench-mode standalone|msprof`
- `--max-concurrency <N>`
- `--show-output`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

### Check Optimization Status

```bash
uv run triton-agent optimize-status --input operators_root
uv run triton-agent optimize-status --input .
uv run triton-agent optimize-status --input operators_root --format markdown
```

Use this command to get a read-only summary of optimization progress across workspaces.
If `--input` already points at one operator workspace, the command summarizes that workspace directly.
It keeps baseline perf files strict, but round `perf.txt` artifacts may include extra metrics such as
`mean_ms` as long as the required `latency-*` entries are still present.
When multiple top-level perf files exist, baseline selection prefers `<original-operator>_perf.txt`,
then `baseline_perf.txt`, then the existing non-`opt_` fallback rule.

`--format markdown` emits a compact table with:

- `名称`
- `Geomean speedup`
- `Total speedup`
- `Verified`
- `Verified Geomean speedup`
- `Verified Total speedup`
- `Notes`

The Markdown table excludes `NO-SESSION` workspaces and sorts rows by name.
Workspaces with optimize artifacts but missing comparable speedup data stay in the table and render those cells as `-`.
The `Verified` column shows `Verified` only when the latest `opt-verify/verify-*/verify-state.json`
for that workspace is a complete successful run with passed test, rerun baseline benchmark,
rerun best benchmark, and compare-perf results. Otherwise it renders `-`.
The verified speedup columns use the same latest successful verify state and stay blank when the
workspace has no verified result.
The `Notes` column uses compact labels such as `best≠log` for computed/logged best-round mismatch
and `warn` for other warnings.

### Verify The Best Round

```bash
uv run triton-agent verify --input .
uv run triton-agent verify --input . --phase test
uv run triton-agent verify --input . --phase bench
```

Use this command after an optimize session when you want to rerun validation for the numeric best round.
The command copies the selected round's operator plus the baseline correctness and benchmark harnesses into
a fresh verification directory, then runs validation there. Existing `baseline/`, `opt-round-*`, top-level
data files, and earlier verification artifacts are not overwritten.

Common options:

- `--phase all|test|bench`: default is `all`.
- `--test-mode standalone|differential`: override the mode recorded in `baseline/state.json`.
- `--bench-mode standalone|msprof`: override the mode recorded in `baseline/state.json`.
- `--remote user@host[:port]`
- `--remote-workdir <path>`
- `--keep-remote-workdir`
- `--verbose`

Each run writes a new directory:

```text
opt-verify/verify-YYYYMMDD-HHMMSS/
```

The directory contains the copied operator, copied harnesses, `test.log`, `bench.log`, generated result or
perf files, `compare-perf.txt` when a benchmark comparison runs, and `verify-state.json`.

### Verify Many Workspaces

```bash
uv run triton-agent verify-batch --input operators_root
uv run triton-agent verify-batch --input operators_root --force-verify
```

Use this command when you want to validate every verifiable optimize workspace under one root.
The command scans immediate child workspaces and:

- reuses the latest `opt-verify/verify-*/verify-state.json` by default
- reruns verification when `--force-verify` is supplied
- skips workspaces that do not have enough baseline or best-round artifacts to run `verify`
- continues after individual workspace failures and reports a non-zero exit code when any rerun fails

Common options:

- `--force-verify`: rerun verification even when a latest verify result already exists.
- `--remote user@host[:port]`
- `--remote-workdir <path>`
- `--keep-remote-workdir`
- `--verbose`

### Optimize In Batch

```bash
uv run triton-agent optimize-batch --input operators_root
uv run triton-agent optimize-batch --input .
```

Common options:

- `--agent codex|opencode|pi|claude|openhands|traecli`
- `--prompt "..."`: append the same extra worker instructions to every workspace optimize run.
- `--test-mode standalone|differential`
- `--bench-mode standalone|msprof`
- `--resume auto|continue|fresh`
- `--reset-optimize`: when used with `--resume fresh`, clear known optimize artifacts for each workspace and reset the batch status file before rerunning
- `--require-analysis`
- `--min-rounds <N>`
- `--no-agent-session`
- `--max-concurrency <N>`: defaults to `1`
- `--show-output`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

Example:

```bash
uv run triton-agent optimize-batch --input operators_root --prompt "Avoid changing numerics unless correctness requires it."
```

Batch rerun behavior:

- `optimize-batch` records explicit completion state in `optimize-batch-status.json` at the batch root.
- A workspace is skipped on rerun only when that file marks it as `completed` and the recorded operator filename still matches.
- Failed workspaces are recorded as `incomplete`, so they remain runnable on the next batch run.
- If the status file is missing or malformed, `optimize-batch` falls back to running all discovered workspaces.
- `--reset-optimize` in batch mode also clears `optimize-batch-status.json` before scheduling workspaces.

## Compare Archived Outputs

Use these commands after you already have archived result or performance files.

### Compare Correctness Results

```bash
uv run triton-agent compare-result \
  --oracle-result abs_result.pt \
  --new-result opt_abs_result.pt
```

Common options:

- `--compare-level strict|balanced|relaxed`
- `--remote user@host[:port]`
- `--remote-workdir <path>`
- `--verbose`

### Compare Performance Results

```bash
uv run triton-agent compare-perf \
  --baseline abs_perf.txt \
  --compare opt_abs_perf.txt
```

The baseline file should stay in the standard `latency-<id>: <float>` format. The compare-side file may
include extra summary fields, which are ignored unless they replace a required latency entry.
The command prints:

- one comparison line per latency id with baseline, compare, and delta
- `Avg improvement` for case-equal percentage improvement
- `Geomean speedup` for benchmark-style speedup aggregation
- `Total speedup` for whole-workload elapsed-time aggregation

## Shared Options

These options appear on multiple commands:

- `--agent`: choose the agent backend for agent-backed generation and optimization commands.
- `--interact`: attach to a live agent session instead of a non-interactive run.
- `--show-output`: stream readable non-interactive agent output in the current terminal.
- `--verbose`: print additional diagnostics.
- `--remote`: run execution and comparison commands through SSH, and pass remote context to generation and optimize workflows.
- `--remote-workdir`: choose the remote working root.
- `--keep-remote-workdir`: keep the remote workspace after `run-test` or `run-bench`.
- `--force-overwrite`: allow generation commands to replace existing generated files.

## Output Conventions

Generated files and archived outputs follow predictable naming based on the operator file:

- tests: typically `test_<op>.py` or `differential_test_<op>.py`
- benchmarks: typically `bench_<op>.py`
- optimized operators: often `opt_<op>.py`
- archived correctness results: typically `<operator>_result.pt`
- archived performance results: typically `<operator>_perf.txt`

## Verification

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```
