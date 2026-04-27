# Batch NPU Affinity Design

## Summary

Add an opt-in batch-only NPU affinity mechanism so concurrent workspace runs can be pinned to distinct Ascend NPU devices through one environment variable. The design must cover the full execution chain, not only the launched code agent process, so local subprocesses and remote SSH-launched benchmark or profiling commands stay on the same assigned device.

## Goal

- Let batch workflows opt into one-device-per-workspace scheduling on multi-NPU machines.
- Keep one workspace bound to one assigned device for the lifetime of that workspace run.
- Propagate the assignment through:
  - CLI-launched code agent backends
  - local run-eval subprocesses
  - remote run-eval commands launched through SSH
- Preserve current behavior completely when the feature is not enabled.

## Non-Goals

- Changing single-workspace commands such as `optimize`, `gen-eval`, or `convert` by default.
- Detecting available NPU devices automatically from the machine.
- Adding prompt-only guidance that asks agents to choose devices themselves.
- Overcommitting one device to multiple concurrent batch workspaces.
- Adding backend-specific affinity behavior that differs by agent vendor.

## Current Problem

The repository already supports bounded workspace concurrency in batch commands such as:

- `optimize-batch`
- `gen-eval-batch`
- `convert-batch`

Today, that concurrency is only a thread-pool scheduling limit. There is no device allocation layer, and no shared environment-injection path for either:

- code agent subprocess launches
- local benchmark/test/profile subprocesses
- remote SSH command execution

As a result, raising `--max-concurrency` on a multi-NPU machine can cause multiple workspaces to contend for the same default visible device.

## User-Facing Control

Introduce one opt-in environment variable:

- `TRITON_AGENT_BATCH_NPU_DEVICES`

Example:

```bash
export TRITON_AGENT_BATCH_NPU_DEVICES=0,1,2,3,4,5,6,7
uv run triton-agent optimize-batch --input operators_root --max-concurrency 8
```

Semantics:

- Unset: disable batch NPU affinity and preserve current behavior.
- Set to a comma-separated list of device identifiers: enable affinity and treat the listed devices as the allowed device pool.
- Whitespace around entries is ignored.
- Empty entries are invalid.
- Duplicate entries are invalid.

The initial implementation should treat device identifiers as opaque strings so the contract can support plain numeric Ascend device IDs without baking in a stricter parser than the runtime needs.

## Scope

The first implementation should support these batch commands:

- `optimize-batch`
- `gen-eval-batch`
- `convert-batch`

The design should also prepare the same mechanism for later reuse by any future concurrent `verify-batch` implementation, but this change does not need to add verify-batch concurrency on its own.

## Design Summary

Add a shared batch-affinity layer that:

1. Parses `TRITON_AGENT_BATCH_NPU_DEVICES`.
2. Creates a bounded device lease pool.
3. Assigns one leased device to each runnable workspace when that workspace actually begins execution.
4. Injects device-specific environment overrides into every local and remote process launched on behalf of that workspace.
5. Releases the lease after the workspace finishes, even on failure.

This should live in the CLI/runtime boundary, not in skill prose or prompt text.

## Device Lease Model

### Why leases instead of static indexing

Static schemes such as `workspace_index % device_count` are not robust once task durations diverge. A long-running workspace could still be using device `0` while a later queued workspace is assigned the same device by index math alone.

### Required behavior

- A workspace acquires a device only when its submitted task starts running.
- A workspace holds that device for the full duration of its batch request.
- The device is released in a `finally` path.
- No two concurrently running workspace tasks may hold the same device lease.

### Concurrency validation

When affinity is enabled:

- `--max-concurrency` must not exceed the number of configured devices.
- If it does, fail explicitly with an actionable message.

This feature should not silently oversubscribe devices through round-robin behavior.

## Environment Contract

For an assigned device `<D>`, inject:

- `ASCEND_RT_VISIBLE_DEVICES=<D>`
- `TRITON_AGENT_ASSIGNED_NPU=<D>`

`ASCEND_RT_VISIBLE_DEVICES` is the actual runtime affinity control. `TRITON_AGENT_ASSIGNED_NPU` is a diagnostic variable for logs, debugging, and possible future script introspection.

The initial design should not require additional affinity variables unless later testing proves Ascend runtime coverage is insufficient.

## Propagation Layers

### 1. Agent launch layer

Extend `AgentRequest` with an `extra_env` field carrying environment overrides for one workspace run.

`AgentRunner` should pass that field into the generic process runner, and the process runner should merge it with the parent environment for:

- interactive runs
- buffered runs
- streaming runs

This ensures the launched agent CLI itself starts with the assigned device constraint.

### 2. Local execution layer

The helper scripts under `skills/triton-npu-run-eval/scripts/` must gain a shared environment-override path for local subprocesses such as:

- local test execution
- local benchmark execution
- local profiling execution
- local remote-support helper commands that currently shell out

This is required because some workflows execute Triton or `msprof` commands directly from repository-owned helper scripts rather than only through the code agent process tree.

### 3. Remote execution layer

Remote commands launched through SSH must receive the same assigned environment explicitly in the remote shell command line.

Conceptually:

```bash
cd <remote_workspace> && ASCEND_RT_VISIBLE_DEVICES=<D> TRITON_AGENT_ASSIGNED_NPU=<D> python3 bench_x.py ...
```

This is required because local process environment does not automatically propagate across SSH boundaries.

## Layering

### Shared affinity module

Add a focused module such as `src/triton_agent/npu_affinity.py` that owns:

- env-var parsing
- validation
- lease-pool creation and acquisition
- conversion from an assigned device to `extra_env`

This module should not know about prompts, skills, or benchmark metadata.

### Batch command modules

Batch orchestration modules should remain responsible for:

- workspace discovery
- request construction
- thread-pool submission
- output prefixing
- result summarization

When affinity is enabled, they additionally acquire a device lease around each workspace run and attach the resolved env overrides to that workspace request.

### Process runners

The generic process runner and run-eval runtime helpers should only know how to accept environment overrides and merge them safely into subprocess launches. They should not parse batch affinity env vars on their own.

## Command Coverage

### `optimize-batch`

This is the highest-priority workflow because a single workspace may run multiple repeated correctness, benchmark, profiling, and repair steps. One workspace should keep the same device through the entire optimize request.

### `gen-eval-batch`

This should use the same affinity mechanism so parallel asset generation plus validation can avoid device contention during generated test or benchmark execution.

### `convert-batch`

This should also use the same mechanism because conversion validation can execute differential tests and other local runtime checks.

### `verify-batch`

Current verify-batch execution is serial. This design should not change that behavior immediately, but the same affinity plumbing should be reusable if verify-batch later adds `--max-concurrency`.

## Failure Semantics

- If affinity is disabled, commands behave exactly as they do today.
- If `TRITON_AGENT_BATCH_NPU_DEVICES` is malformed, fail explicitly before launching any workspace tasks.
- If `--max-concurrency` exceeds configured device count while affinity is enabled, fail explicitly before launching any workspace tasks.
- If one workspace run raises or returns failure, release its device lease and continue processing other workspaces using existing batch semantics.
- Device release must happen even when request construction or execution throws unexpectedly after the lease is acquired.

## Logging And Diagnostics

Verbose mode should include enough information to answer:

- whether affinity was enabled
- which device pool was configured
- which workspace received which device

This should be additive diagnostic output only. The standard non-verbose user-facing batch summary should remain compact.

## Files Expected To Change

- `src/triton_agent/models.py`
  - add per-request environment overrides
- `src/triton_agent/npu_affinity.py`
  - new shared affinity parsing and lease logic
- `src/triton_agent/process_runner.py`
  - accept explicit env overrides for subprocess launches
- `src/triton_agent/backends/base.py`
  - pass request env overrides into process execution
- `src/triton_agent/optimize/batch.py`
  - acquire and release device leases per workspace
- `src/triton_agent/generation/batch.py`
  - acquire and release device leases per workspace
- `src/triton_agent/convert/batch.py`
  - acquire and release device leases per workspace
- `skills/triton-npu-run-eval/scripts/run_runtime.py`
  - add shared env override support for local and remote command execution
- tests covering parsing, lease behavior, subprocess env propagation, and batch integration
- `README.md`
  - document the opt-in env variable and the concurrency constraint

## Testing Strategy

Add or update tests for:

1. Parsing a valid `TRITON_AGENT_BATCH_NPU_DEVICES` list.
2. Rejecting empty, duplicate, or malformed device entries.
3. Rejecting affinity-enabled runs where `--max-concurrency` exceeds device count.
4. Assigning distinct devices to concurrently running batch workspaces.
5. Releasing leases after success.
6. Releasing leases after failures or exceptions.
7. Passing `extra_env` into backend-launched subprocesses.
8. Passing env overrides into local run-eval subprocesses.
9. Prefixing remote SSH command strings with the assigned env overrides.
10. Preserving current behavior when the affinity env var is unset.

## Risks And Mitigations

### Risk: only the agent process is pinned

If env propagation stops at the code agent boundary, benchmark and profiling subprocesses may still contend for devices.

Mitigation: implement env override support in both the backend process runner and the run-eval runtime helper layer.

### Risk: remote runs ignore local affinity settings

Remote SSH commands do not inherit the local process environment automatically.

Mitigation: explicitly prefix remote command strings with the assigned environment variables.

### Risk: lease leaks reduce available capacity

If a workspace crashes after acquiring a device but before normal completion, that device could remain unavailable.

Mitigation: require acquisition and release to be paired in `try/finally` blocks around workspace execution.

### Risk: hidden oversubscription

If affinity silently reuses devices when concurrency exceeds pool size, users may think they have isolated runs when they do not.

Mitigation: fail early when requested concurrency exceeds configured device count.

## Open Questions Resolved

- Should the implementation rely only on prompt instructions to choose devices? No.
- Is adding env overrides only to code agent launches sufficient? No.
- Should the feature be opt-in and batch-only at first? Yes.
- Should the initial implementation oversubscribe devices with round-robin assignment? No.
