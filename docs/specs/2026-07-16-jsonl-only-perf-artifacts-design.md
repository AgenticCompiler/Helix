# JSONL-Only Perf Artifacts Design

## User-Visible Behavior

Performance artifact readers accept JSONL records only. A file named `perf.txt`
may still contain JSONL, so existing artifact paths and state contracts do not
change. Historical line-plus-comment perf content such as
`latency-case-1: 1.0` is rejected with an actionable JSONL-only error.

## Implementation

Keep the existing JSONL parser as the sole parser behind the public perf
artifact read APIs. Remove the text-format detection branches and their
raw-op-statistic, latency-error, and text rendering helpers. Required-ID
matching remains unchanged for JSONL records.

## Verification

Cover rejection of a legacy text record and retain JSONL parser coverage. Migrate
all perf-consuming test fixtures to JSONL, then run the full suite and the
required strict Pyright check for the modified skill script.
