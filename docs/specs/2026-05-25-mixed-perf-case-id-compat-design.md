# Mixed Perf Case-ID Compatibility Design

## Summary

Restore cross-format `compare-perf` compatibility when a legacy `msprof` text perf artifact uses `latency-case-<N>` ids and a new JSONL perf artifact stores the same cases as numeric `case_label` values such as `"1"`.

## Problem

Legacy `msprof` perf artifacts expose case ids as `latency-case-<N>`. The new JSONL perf writer currently stores `case_label` as the raw numeric index, which the compatibility parser derives as `latency-<N>`. Each file parses on its own, but mixed old/new comparisons fail because required latency ids no longer match.

## Goals

- Let existing legacy baseline perf artifacts compare against already-generated JSONL perf artifacts.
- Restore stable future `msprof` JSONL output so newly generated artifacts keep the legacy public case-id contract.
- Keep standalone benchmark case ids unchanged.

## Non-Goals

- Do not redesign the public `compare-perf` CLI.
- Do not migrate historical perf artifacts in place.
- Do not change standalone benchmark case-id semantics.

## Design

### Consumer compatibility

Teach required-id parsing in `skills/triton-npu-run-eval/scripts/perf_artifacts.py` to accept a legacy `latency-case-<N>` requirement when a JSONL `msprof` record exposes the same logical case as numeric `case_label: "N"`.

The compatibility should stay narrow:

- only required-id matching changes
- exact ids still win first
- standalone ids such as `latency-foo` remain unchanged

This fixes the user-visible failure mode in `compare-perf`, plus status and verify flows that already parse compare-side perf artifacts through required-id helpers.

### Future writer stability

Update `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py` so new JSONL `msprof` records write `case_label` as `case-<N>` instead of bare numeric text. That keeps the derived JSONL latency id aligned with the long-standing `latency-case-<N>` contract.

## Verification

- Add a failing regression test for `compare_perf_files()` with:
  - legacy baseline text: `latency-case-1`
  - JSONL compare: `{"case_label":"1", ...}`
- Keep the existing JSONL parser tests green.
- Run focused comparison tests plus the strict skill-script pyright check for `perf_artifacts.py`.
