# Batch Concurrency Keyword Design

## Goal

Rename the batch CLI flag `--max-concurrency` to `--concurrency`, and let the affinity-aware batch commands accept a symbolic `max` value that expands to the largest legal concurrency derived from the configured NPU pool.

## Scope

This change applies to the current batch commands that expose a concurrency flag:

- `gen-eval-batch`
- `convert-batch`
- `optimize-batch`
- `log-check-batch`

Behavior differs by command family:

- `gen-eval-batch`, `convert-batch`, and `optimize-batch` accept either a positive integer or the literal `max`.
- `log-check-batch` is renamed to `--concurrency` for CLI consistency, but it remains numeric-only because it has no NPU-capacity concept.

The old `--max-concurrency` flag is removed rather than kept as an alias.

## User-Facing Behavior

For the affinity-aware batch commands:

- `--concurrency 4` keeps the current numeric behavior.
- `--concurrency max` resolves to `len(HELIX_BATCH_NPU_DEVICES) * HELIX_BATCH_WORKERS_PER_NPU`.
- when omitted, `--concurrency` defaults to `1`
- If `HELIX_BATCH_NPU_DEVICES` is unset, `--concurrency max` fails with a short actionable error because there is no defined device pool to expand.
- Numeric values must still be at least `1`.
- Numeric values must still not exceed the effective batch affinity capacity when affinity is enabled.

For `log-check-batch`:

- `--concurrency <N>` replaces `--max-concurrency <N>`.
- The value must be a positive integer.
- `max` is rejected at parse time.

## Design

### CLI parsing

Add two parser helpers in `src/helix/cli.py`:

- one for numeric-or-`max` concurrency values
- one for numeric-only concurrency values

Use those helpers when adding the batch concurrency argument so parser results stay strongly typed:

- positive integer input parses to `int`
- `max` parses to the string literal `"max"`

This keeps downstream command handlers simple and avoids hand-parsing raw strings in multiple places.

### Affinity-aware concurrency resolution

Add a shared helper in `src/helix/npu_affinity.py` that resolves the requested batch concurrency into a concrete integer for the three affinity-aware batch commands.

Resolution rules:

1. integer input returns unchanged after the existing lower-bound and capacity checks
2. `max` computes the effective capacity from the configured devices and workers-per-device
3. `max` without `HELIX_BATCH_NPU_DEVICES` raises a clear `ValueError`

The existing capacity check remains the source of truth for numeric oversubscription.

### Command handlers

Update the affected command handlers to consume `args.concurrency` instead of `args.max_concurrency`.

- `gen-eval-batch`
- `convert-batch`
- `optimize-batch`
- `log-check-batch`

Only the first three call the new affinity-aware resolver before invoking the batch runtime.

### Runtime code

The batch runtime helpers can continue taking an integer `max_concurrency` parameter internally. The rename is primarily a CLI contract change plus one new symbolic value at the command layer.

This keeps the implementation small and avoids unnecessary churn in the executor layer and its tests.

## Error Handling

- `--concurrency 0` keeps failing with a short lower-bound error.
- `--concurrency max` without `HELIX_BATCH_NPU_DEVICES` fails before any workspace launch.
- malformed `HELIX_BATCH_WORKERS_PER_NPU` continues to fail through the existing parser.
- oversubscription errors should now reference `--concurrency` instead of `--max-concurrency`.

## Tests

Add or update focused coverage for:

- parser defaults under the new `args.concurrency` attribute
- parser acceptance of `--concurrency max` on affinity-aware batch commands
- parser rejection of `--concurrency max` for `log-check-batch`
- command-level resolution of `max` into effective capacity
- failure when `max` is requested without `HELIX_BATCH_NPU_DEVICES`
- existing batch affinity capacity errors with the renamed flag text
