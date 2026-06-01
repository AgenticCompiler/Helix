# Batch Workers Per NPU Design

## Summary

Extend the existing batch NPU affinity model with an opt-in environment variable that allows more than one concurrent batch workspace to share the same configured NPU device. The new control must compose with `TRITON_AGENT_BATCH_NPU_DEVICES`, stay batch-only, and apply consistently everywhere the existing batch NPU device list is already supported.

## Goal

- Preserve the current `TRITON_AGENT_BATCH_NPU_DEVICES` contract as the source of truth for which NPUs may be used by batch commands.
- Let users opt into bounded sharing by declaring how many concurrent batch workers each configured NPU may host.
- Keep the runtime affinity mechanism unchanged at the process boundary by continuing to inject `ASCEND_RT_VISIBLE_DEVICES=<device>`.
- Apply the same behavior to every current batch command that already supports `TRITON_AGENT_BATCH_NPU_DEVICES`.

## Non-Goals

- Changing single-workspace commands such as `optimize`, `convert`, or `gen-eval`.
- Enabling any NPU-sharing behavior when `TRITON_AGENT_BATCH_NPU_DEVICES` is unset.
- Adding per-device custom capacities such as `0:1,1:2`.
- Detecting device capacity automatically from hardware state.
- Introducing different sharing semantics for different batch commands.

## Current Behavior

Today, `TRITON_AGENT_BATCH_NPU_DEVICES` enables a one-device-per-workspace lease model for:

- `optimize-batch`
- `convert-batch`
- `gen-eval-batch`

The current affinity pool treats each configured device as one leaseable slot. As a result:

- `--max-concurrency` must not exceed the number of configured devices.
- No two concurrently running batch workspaces may receive the same device.
- Each workspace receives one `ASCEND_RT_VISIBLE_DEVICES` value for the lifetime of its request.

This behavior is implemented centrally in `src/triton_agent/npu_affinity.py` and reused by the batch command modules.

## User-Facing Controls

Keep the existing device-list variable:

- `TRITON_AGENT_BATCH_NPU_DEVICES`

Add one new environment variable:

- `TRITON_AGENT_BATCH_WORKERS_PER_NPU`

Example:

```bash
export TRITON_AGENT_BATCH_NPU_DEVICES=0,1
export TRITON_AGENT_BATCH_WORKERS_PER_NPU=2
uv run triton-agent optimize-batch --input operators_root --max-concurrency 4
```

Semantics:

- If `TRITON_AGENT_BATCH_NPU_DEVICES` is unset, `TRITON_AGENT_BATCH_WORKERS_PER_NPU` is ignored.
- If `TRITON_AGENT_BATCH_NPU_DEVICES` is set and `TRITON_AGENT_BATCH_WORKERS_PER_NPU` is unset, the effective value is `1`.
- `TRITON_AGENT_BATCH_WORKERS_PER_NPU` must be a positive integer.
- `0`, negative values, empty values, and non-integer values fail explicitly with actionable validation errors.

## Desired Behavior

When both variables are set:

- The configured device list still defines the only allowed devices.
- Each configured device contributes `workers_per_npu` concurrent scheduling slots.
- Multiple concurrently running workspaces may receive the same device identifier when capacity for that device has not been exhausted.
- Each individual workspace still receives exactly one `ASCEND_RT_VISIBLE_DEVICES` value.

Example expansions:

- `devices=0,1` and `workers_per_npu=1` => capacity `2`, effective slots `["0", "1"]`
- `devices=0,1` and `workers_per_npu=2` => capacity `4`, effective slots `["0", "0", "1", "1"]`
- `devices=0,3-4` and `workers_per_npu=3` => capacity `9`, effective slots `["0", "0", "0", "3", "3", "3", "4", "4", "4"]`

The slot expansion above is descriptive, not a user-visible API. It explains the intended lease behavior while keeping the process-level environment contract unchanged.

## Design Summary

Implement the feature by extending the shared batch affinity layer rather than special-casing individual commands:

1. Parse `TRITON_AGENT_BATCH_NPU_DEVICES` exactly as today.
2. Parse `TRITON_AGENT_BATCH_WORKERS_PER_NPU` as a positive integer, but only when the device list is enabled.
3. Expand the configured device tuple into a slot tuple that repeats each device `workers_per_npu` times.
4. Reuse the existing lease-pool pattern on top of those expanded slots.
5. Keep downstream command modules unchanged apart from calling the shared helper APIs.

This approach intentionally keeps the external process environment unchanged: each launched workspace still receives a single `ASCEND_RT_VISIBLE_DEVICES=<device>` binding.

## Shared Affinity Model

### Why slot expansion

The current pool model already expresses the desired lifecycle well:

- acquire one lease when a workspace starts
- hold that lease for the full workspace run
- release the lease in a `finally` path

Expanding each configured device into multiple identical slots lets the project preserve that existing lifecycle without introducing a more complex per-device counter structure.

### Required behavior

- A workspace acquires one slot only when execution begins.
- A workspace holds one slot for the full duration of its batch request.
- Releasing one slot for device `0` makes that device available to another waiting workspace, up to the configured per-device capacity.
- Two or more concurrent workspaces may hold distinct slots that map to the same device identifier when the configured worker count allows it.

## Parsing And Validation

Add shared parsing and validation in `src/triton_agent/npu_affinity.py` for:

- the existing device list
- the new workers-per-device integer
- the derived effective slot pool

Recommended helper shape:

- `configured_batch_npu_devices()`
- `configured_batch_workers_per_npu()`
- `configured_batch_npu_slots()`

Behavior:

- `configured_batch_workers_per_npu()` returns `1` when the env var is unset.
- `configured_batch_npu_slots()` returns `None` when `TRITON_AGENT_BATCH_NPU_DEVICES` is unset.
- `configured_batch_npu_slots()` returns the expanded slot tuple when devices are configured.

This keeps batch callers from having to duplicate the “ignore workers-per-npu unless devices are configured” rule.

## Capacity Rules

When `TRITON_AGENT_BATCH_NPU_DEVICES` is unset:

- preserve current behavior
- do not validate `TRITON_AGENT_BATCH_WORKERS_PER_NPU`
- do not inject any affinity env override

When `TRITON_AGENT_BATCH_NPU_DEVICES` is set:

- effective affinity capacity is `len(devices) * workers_per_npu`
- `--max-concurrency` must not exceed that effective capacity
- validation errors should mention both environment variables so the user knows how capacity was derived

Example:

- `TRITON_AGENT_BATCH_NPU_DEVICES=0,1`
- `TRITON_AGENT_BATCH_WORKERS_PER_NPU=2`
- maximum legal `--max-concurrency` is `4`

## Command Coverage

Because the current device-list affinity is already shared, the new workers-per-device control should automatically apply to the same supported commands:

- `optimize-batch`
- `convert-batch`
- `gen-eval-batch`

No command-specific opt-in flag is needed. The contract is environment-driven and uniform across all current batch-affinity users.

## Environment Contract

Do not change the process-level affinity environment:

- Each launched workspace still receives exactly one `ASCEND_RT_VISIBLE_DEVICES=<device>` value.
- Sharing is expressed only by allowing more than one concurrent workspace lease for the same device identifier.

This keeps the feature compatible with existing backend launch, local subprocess, and remote subprocess env-propagation paths.

## Failure Semantics

- If `TRITON_AGENT_BATCH_NPU_DEVICES` is unset, batch commands behave exactly as they do today, regardless of whether `TRITON_AGENT_BATCH_WORKERS_PER_NPU` is set.
- If `TRITON_AGENT_BATCH_NPU_DEVICES` is set and `TRITON_AGENT_BATCH_WORKERS_PER_NPU` is malformed, fail before launching any workspace tasks.
- If `--max-concurrency` exceeds effective capacity, fail before launching any workspace tasks.
- If a workspace run fails after acquiring a slot, release the slot and continue existing batch result handling.

## Documentation Changes

Update user-facing docs to explain:

- the new variable name
- that it only matters when `TRITON_AGENT_BATCH_NPU_DEVICES` is set
- that effective capacity is `device_count * workers_per_npu`
- that all current batch-affinity commands share the same behavior

The README environment-variable table, batch affinity section, and command examples should all use the same terminology.

## Testing And Verification

Add or update tests for:

- parsing valid workers-per-device values
- rejecting invalid workers-per-device values
- ignoring the new variable when the device list is unset
- expanded capacity validation
- allowing repeated `ASCEND_RT_VISIBLE_DEVICES` assignments when sharing is enabled
- keeping the current one-slot-per-device behavior when the new variable is unset

Targeted coverage should include:

- `tests/test_npu_affinity.py`
- `tests/test_optimize_runtime.py`
- `tests/test_convert_commands.py`
- `tests/test_generation_batch.py`

Repository verification should include:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

## Files Expected To Change

- `src/triton_agent/npu_affinity.py`
- `src/triton_agent/optimize/batch.py`
- `src/triton_agent/convert/batch.py`
- `src/triton_agent/generation/batch.py`
- `src/triton_agent/cli.py`
- `README.md`
- `tests/test_npu_affinity.py`
- `tests/test_optimize_runtime.py`
- `tests/test_convert_commands.py`
- `tests/test_generation_batch.py`
