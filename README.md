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
- `--agent codex|opencode|pi|claude`: choose the backend.
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

- `--agent codex|opencode|pi|claude`
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
- `--agent codex|opencode|pi|claude`
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
- `--agent codex|opencode|pi|claude`
- `--test-mode standalone|differential`: default is `differential`
- `--bench-mode standalone|msprof`: default is `standalone`
- `--resume auto|continue|fresh`: default is `auto`
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
```

Resume modes:

- `auto`: continue only when there is a complete existing optimize session; otherwise start fresh or fail if the workspace is incomplete.
- `continue`: require an existing resumable optimize session.
- `fresh`: require a clean workspace with no existing optimize artifacts.

Optimize behavior:

- Reuse existing test and benchmark harnesses when they already exist in the workspace.
- Generate missing harnesses only when the required validation artifact is absent.
- Run optimize as explicit worker rounds with a supervisor audit between rounds instead of relying on one unconstrained agent pass.
- Keep the shared workspace guidance role-neutral; worker versus supervisor role assignment comes from the launch prompt and role brief for that invocation.
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

- `--agent codex|opencode|pi|claude`
- `--test-mode standalone|differential`
- `--bench-mode standalone|msprof`
- `--max-concurrency <N>`
- `--show-output`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

### Check Optimization Status

```bash
uv run triton-agent optimize-status --input operators_root
uv run triton-agent optimize-status --input operators_root --format markdown
```

Use this command to get a read-only summary of optimization progress across workspaces.
It keeps baseline perf files strict, but round `perf.txt` artifacts may include extra metrics such as
`mean_ms` as long as the required `latency-*` entries are still present.

`--format markdown` emits a compact table with:

- `名称`
- `Geomean speedup`
- `Total speedup`

The Markdown table excludes `NO-SESSION` workspaces. Workspaces with optimize artifacts but missing
comparable speedup data stay in the table and render those cells as `-`.

### Optimize In Batch

```bash
uv run triton-agent optimize-batch --input operators_root
uv run triton-agent optimize-batch --input .
```

Common options:

- `--agent codex|opencode|pi|claude`
- `--test-mode standalone|differential`
- `--bench-mode standalone|msprof`
- `--resume auto|continue|fresh`
- `--require-analysis`
- `--min-rounds <N>`
- `--no-agent-session`
- `--max-concurrency <N>`
- `--show-output`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

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
