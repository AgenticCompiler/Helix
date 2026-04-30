# triton-agent

`triton-agent` is a CLI for generating, running, and optimizing Triton Ascend NPU operator workflows with code agents and local skills.

This README is organized by task so you can quickly find the right command for the job.

## Command Map

- `gen-test`: generate a correctness test for one operator.
- `run-test`: run an existing generated test.
- `gen-eval`: generate both test and benchmark assets for one operator.
- `gen-eval-batch`: generate evaluation assets for many operator workspaces.
- `convert`: convert one PyTorch operator into a Triton NPU-backed PyTorch operator and validate it with differential testing.
- `convert-batch`: convert many operator workspaces.
- `gen-bench`: generate a benchmark for one operator.
- `run-bench`: run an existing generated benchmark.
- `optimize`: optimize one operator.
- `status`: summarize optimization progress across many workspaces.
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

uv run triton-agent convert --input a.py
uv run triton-agent convert-batch --input operators_root

uv run triton-agent gen-bench --input a.py
uv run triton-agent run-bench --bench-file bench_a.py --operator-file a.py

uv run triton-agent optimize --input a.py
```

For batch workflows, point `--input` at either a directory whose immediate child directories are operator workspaces, or a single operator workspace directory:

```bash
uv run triton-agent gen-eval-batch --input operators_root
uv run triton-agent gen-eval-batch --input .
uv run triton-agent status --input operators_root
uv run triton-agent status --input operators_root --format markdown
uv run triton-agent verify --input .
uv run triton-agent verify-batch --input operators_root
uv run triton-agent optimize-batch --input operators_root
```

## Runtime Environment Variables

These are the environment variables that `triton-agent` reads directly at runtime.

| Variable | Required | Used by | Purpose |
| --- | --- | --- | --- |
| `TRITON_AGENT_HOME` | No | `optimize`, `optimize-batch` with `--enable-compiler-source-analysis` | Overrides the default Triton Agent home directory. The compiler-source checkout is stored under `<TRITON_AGENT_HOME>/compiler-sources/AscendNPU-IR/` instead of `~/.triton-agent/compiler-sources/AscendNPU-IR/`. |
| `TRITON_AGENT_BATCH_NPU_DEVICES` | No | `gen-eval-batch`, `convert-batch`, `optimize-batch` | Comma-separated Ascend device list that also supports inclusive numeric ranges such as `0,3-5,8-9`. When set, concurrent batch workspaces are pinned to distinct devices, and `--max-concurrency` must not exceed the number of configured devices. |
| `TRITON_AGENT_CODE_AGENT_MAX_RETRIES` | No | Agent-backed commands | Non-negative integer retry budget for transient code-agent failures such as rate limits. Default is `2`. Set `0` to disable retries. |
| `TRITON_AGENT_MSPROF_OUTPUT_DIR` | No | Local `msprof` benchmark runs | Preserves local `msprof` output directories under the given root instead of using auto-cleaned temporary directories. Useful when you want to inspect profiler artifacts after `run-bench`, `verify`, or optimize validation in `msprof` mode. |
| `LLM_API_KEY` | Only for `--agent openhands` | OpenHands backend | API key forwarded to the OpenHands SDK LLM client. |
| `LLM_MODEL` | Only for `--agent openhands` | OpenHands backend | Model name passed to the OpenHands SDK LLM client. |
| `LLM_BASE_URL` | No | OpenHands backend | Optional custom base URL for the OpenHands SDK LLM client. |

Examples:

```bash
export TRITON_AGENT_BATCH_NPU_DEVICES=0,3-5,8-9
export TRITON_AGENT_CODE_AGENT_MAX_RETRIES=4
export TRITON_AGENT_HOME=$HOME/.triton-agent
uv run triton-agent optimize-batch --input operators_root --max-concurrency 4
```

```bash
export LLM_API_KEY=...
export LLM_MODEL=openai/gpt-4.1
export LLM_BASE_URL=https://api.example.com/v1
uv run triton-agent gen-test --input a.py --agent openhands
```

### Environment Variables Exported By `triton-agent`

These variables are normally set by `triton-agent` for child processes. You usually do not need to export them yourself:

- `ASCEND_RT_VISIBLE_DEVICES`: set for each batch workspace when `TRITON_AGENT_BATCH_NPU_DEVICES` is configured.
- `TRITON_AGENT_ASSIGNED_NPU`: companion variable that records the device chosen for the current workspace.

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

## Convert PyTorch Operators

Use `convert` when you want a new Triton NPU-backed PyTorch operator file instead of an in-place optimize round.

```bash
uv run triton-agent convert --input a.py
```

You may also point `--input` at a single operator workspace directory when that directory contains exactly one candidate operator file, for example `uv run triton-agent convert --input .`.

What it is for:

- converting one source PyTorch operator into a Triton NPU-backed PyTorch operator
- preserving the input file's trailing input-helper block in the converted output
- validating the converted operator through differential correctness validation against the original operator

Common options:

- `--output triton_a.py`: write to a specific converted-operator path.
- `--agent codex|opencode|pi|claude|openhands|traecli`
- `--test-mode differential`
- `--interact`
- `--show-output`
- `--force-overwrite`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

Behavior:

- The original input operator file is treated as source material and differential correctness oracle and must not be executed by this workflow.
- The converted output defaults to `triton_<origin-name>.py`.
- The input file's trailing input-helper block should remain available in the converted output.
- The workflow generates and executes a differential test for the converted output before finishing.
- When `--input` is a workspace directory, staged skills and agent cwd are rooted at that workspace.

Example:

```bash
uv run triton-agent convert --input a.py --output triton_a.py
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

For `standalone` benchmarks:

- the benchmark file is import-only and exports `build_operator_api(operator_module)` plus `build_standalone_bench_cases(operator_api)`
- `run-bench` profiles each declared case with `torch_npu.profiler`
- `profile-bench` requires `--case-id <id>` for standalone profiling

For `msprof` benchmarks:

- `run-bench` aggregates the stable-order union of benchmark metadata kernels and `@triton.jit` kernels discovered from the runtime operator file.
- a failed benchmark case does not stop later cases from running
- the generated perf file is still written and includes `# latency-error-case-*` comments for failed cases
- `profile-bench` should still be invoked with an explicit `--kernel-name <name>` unless the metadata resolves to exactly one kernel

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
- `--bench-mode standalone|msprof`: default is `standalone`. Sets the benchmark mode for fresh runs. With `--resume auto`, resumable workspaces keep the benchmark mode recorded in their existing benchmark harness.
- `--resume auto|continue|fresh`: default is `auto`
- `--reset-optimize`: only valid with `--resume fresh`; remove known optimize-session artifacts before starting a new run while keeping reusable test and benchmark harnesses.
- `--enable-compiler-source-analysis`: allow the optimize agent to use compiler source as an escalation after benchmark, profiler, and IR evidence.
- `--enable-cann-ext-api`: allow A5-only CANN Triton extension API optimization patterns during optimize runs.
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
uv run triton-agent optimize --input a.py --enable-compiler-source-analysis
uv run triton-agent optimize --input a.py --enable-cann-ext-api --target-chip A5
uv run triton-agent optimize --input a.py --prompt "Prioritize memory-coalescing improvements."
```

Compiler source analysis is opt-in. When enabled, the CLI prepares a shallow AscendNPU-IR checkout under `~/.triton-agent/compiler-sources/AscendNPU-IR/` before launching the agent, using the configured Triton Agent home when `TRITON_AGENT_HOME` is set. The launched agent receives only the local path and commit, treats the checkout as read-only, and must not clone, fetch, pull, or modify compiler source. This option enables an escalation path for difficult compiler-side explanations; it does not require compiler-source analysis in every round.

CANN extension API pattern access is also opt-in. When `--enable-cann-ext-api` is set, optimize stages a dedicated skill with specialized CANN Triton extension API guidance, including `sub_vec_id()`-based rewrite patterns. This option is valid only with `--target-chip A5`.

Resume modes:

- `auto`: continue only when there is a complete existing optimize session; otherwise start fresh or fail if the workspace is incomplete.
- `continue`: require an existing resumable optimize session.
- `fresh`: require a clean workspace with no existing optimize artifacts.
- `fresh` + `--reset-optimize`: delete known optimize-session artifacts first, but keep reusable generated test and benchmark harnesses.

Optimize behavior:

- Establish or reuse a canonical `baseline/` directory before treating `opt-round-1` as the first optimization round.
- If `baseline/` is missing or invalid, baseline preparation is handled by `triton-npu-prepare-optimize-baseline` before round work begins.
- Every optimize run follows the default layered analysis ladder: pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation.
- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.
- Use compiler source only as the deepest escalation, and only when `--enable-compiler-source-analysis` is set.
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

Use the batch commands when `--input` points to a directory of operator workspaces. `gen-eval-batch`, `convert-batch`, and `optimize-batch` can also accept one operator workspace directory directly.

### Batch NPU Affinity

Set `TRITON_AGENT_BATCH_NPU_DEVICES` to a comma-separated device list when you want concurrent batch workspaces to stay on distinct Ascend NPUs. The value supports explicit IDs and inclusive numeric ranges:

```bash
export TRITON_AGENT_BATCH_NPU_DEVICES=0,3-5,8-9
uv run triton-agent optimize-batch --input operators_root --max-concurrency 4
```

When this variable is set:

- `gen-eval-batch`, `convert-batch`, and `optimize-batch` assign one configured device per running workspace.
- `--max-concurrency` must not exceed the number of configured devices.
- the assigned device is exported as `ASCEND_RT_VISIBLE_DEVICES` for launched workspace processes.

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

### Convert Operators In Batch

```bash
uv run triton-agent convert-batch --input operators_root
```

Common options:

- `--agent codex|opencode|pi|claude|openhands|traecli`
- `--test-mode differential`
- `--max-concurrency <N>`
- `--show-output`
- `--remote user@host[:port]`
- `--remote-workdir <path>`

### Check Optimization Status

```bash
uv run triton-agent status --input operators_root
uv run triton-agent status --input .
uv run triton-agent status --input operators_root --format markdown
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
- `--bench-mode standalone|msprof`: sets the benchmark mode for fresh workspaces. With `--resume auto`, resumable workspaces keep the benchmark mode recorded in their existing benchmark harness.
- `--resume auto|continue|fresh`
- `--reset-optimize`: when used with `--resume fresh`, clear known optimize artifacts for each workspace and reset the batch status file before rerunning
- `--enable-compiler-source-analysis`
- `--enable-cann-ext-api`
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
