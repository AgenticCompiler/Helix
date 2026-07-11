# Status Metric Source Design

## Summary

Add an optional `-m` / `--metric-source` flag to `helix status` so users can force one comparison basis for status calculations:

- `auto`
- `kernel`
- `total-op`

When the flag is omitted, `status` must preserve current behavior and keep using each round's recorded effective metric source.

## User-Visible Behavior

- `helix status --metric-source kernel` recomputes best-view and trend-view speedups from kernel latency only.
- `helix status --metric-source total-op` recomputes best-view and trend-view speedups from total-op timing only.
- `helix status --metric-source auto` recomputes speedups using the existing kernel-first fallback behavior from shared perf parsing.
- Omitting `--metric-source` preserves the current round-recorded basis, so existing `status` output does not change.

`status` should not accept `--metric-source all`.

Reason:

- `status` renders one status summary per invocation, not separate metric sections like `compare-perf`.
- Supporting `all` would require a larger output redesign and is out of scope for this change.

## Scope

In scope:

- the repository CLI parser for `status`
- the `status` command handler argument forwarding
- status core helpers for single-workspace and batch scans
- README examples and option wording

Out of scope:

- status renderer shape changes
- new JSON fields for metric-source override reporting
- changing default status semantics
- support for `--metric-source all`

## Design

Parser:

- Register `-m` / `--metric-source` on `status`
- Accept `auto|kernel|total-op`
- Default to `None`, not `"auto"`

Runtime:

- `handle_status(...)` forwards the optional override into both:
  - single-workspace inspection
  - multi-workspace scanning
- `inspect_optimize_status_workspace(...)` chooses the metric source per round as:
  1. explicit CLI override, when provided
  2. otherwise the current recorded round effective metric source logic

This keeps renderers unchanged because they already consume computed round/workspace speedups without caring how the source was selected.

## Testing

- parser tests for:
  - accepting `--metric-source`
  - accepting `-m`
  - rejecting `all`
  - defaulting to `None`
- status-core tests proving an explicit override can change best-round selection
- CLI integration test proving batch `status` output reflects the override
