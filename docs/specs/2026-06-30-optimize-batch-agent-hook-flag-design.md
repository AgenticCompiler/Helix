# Optimize Batch Agent Hook Flag Design

## Goal

Make `triton-agent optimize-batch` accept the same agent-hook enable flags as
`triton-agent optimize` so batch optimize runs can opt into request-scoped hook
guard behavior without falling back to single-workspace invocations.

## User-Visible Semantics

- `optimize-batch` accepts both `--enable-agent-hook` and
  `--enable-agent-hooks`.
- The parsed flag maps to the existing `enable_agent_hooks` optimize run option.
- Each workspace optimize request launched by `optimize-batch` keeps using the
  existing hook staging behavior already implemented for optimize requests.
- Without the flag, `optimize-batch` behavior remains unchanged.

## Problem

The batch optimize runtime already builds per-workspace optimize requests from
`OptimizeRunOptions`, and those requests already support
`enable_agent_hooks=True`. The user-facing gap is earlier in the stack:
`build_parser()` only registers the hook flag for `CommandKind.OPTIMIZE`, so
`optimize-batch --enable-agent-hook` fails during argument parsing before the
option can reach the shared optimize request path.

## Scope

In scope:

- parser support for the singular and plural hook flags on `optimize-batch`
- regression coverage for parser behavior
- README updates for batch optimize and shared hook-guard documentation

Out of scope:

- changing hook staging behavior
- changing default hook policy behavior
- adding hook support to non-optimize batch commands

## Design

Expose the existing hook flag for both optimize command kinds by extending the
parser condition from only `OPTIMIZE` to both `OPTIMIZE` and
`OPTIMIZE_BATCH`. Reuse the existing `dest="enable_agent_hooks"` wiring so
`optimize_run_options_from_args()` and downstream batch request construction do
not need new data plumbing.

Cover the change with one CLI parser regression test for `optimize-batch`, then
document the new option in the batch optimize README section and broaden the
shared hook-guard wording from optimize-only to optimize and optimize-batch.
