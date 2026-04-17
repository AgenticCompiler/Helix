# Optimize Compare-Perf Authority Design

## Summary

- Treat `compare-perf` as the only authoritative source for optimize-round performance deltas and speedup metrics.
- Prevent optimize agents from hand-computing percentage improvements or speedups from raw perf files.
- Enforce this rule in both agent-facing guidance and CLI-side round gating so a prompt miss does not silently corrupt optimize conclusions.

## Problem

- The optimize workflow already requires comparable perf artifacts, but current worker guidance does not explicitly say that performance conclusions must come from `compare-perf`.
- As a result, a code agent may read raw `*_perf.txt` files, derive its own arithmetic, and write incorrect improvement numbers into round summaries or `opt-note.md`.
- When that happens, the optimize session can preserve and promote wrong benchmark conclusions even if the underlying perf artifacts are present.

## Goals

- Make the `compare-perf` helper the mandatory performance-summary path for optimize rounds.
- Tell optimize workers and supervisors that benchmark wins must be justified with `compare-perf`, not ad-hoc arithmetic.
- Extend the round contract so the CLI can reject rounds whose claimed performance summary did not come from `compare-perf`.
- Keep the implementation small and additive without moving optimize-domain reasoning out of the skill.

## Non-Goals

- Do not redesign the `compare-perf` output format.
- Do not move benchmark execution or comparison logic out of the existing triton-npu-run-eval helper scripts.
- Do not require the CLI to parse free-form `summary.md` prose for every claimed number.

## Design

### Guidance And Prompt Changes

- Update optimize worker guidance, unsupervised optimize prompts, and shared role briefs to state:
  - run `compare-perf` after baseline and candidate perf artifacts exist
  - use `compare-perf` output as the only source for `Avg improvement`, `Geomean speedup`, and `Total speedup`
  - do not hand-calculate or restate benchmark deltas from raw perf files
- Update supervisor guidance so audit passes block or repair only metadata derived from an existing `compare-perf` result.

### Round Contract Changes

Add one required `round-state.json` field:

- `perf_summary_source`

Allowed value for a passing benchmark round:

- `compare-perf`

This field records the origin of the round's performance conclusion, not merely the existence of a perf artifact.

### Gate Changes

- If `benchmark_status != "passed"`, keep the existing gate behavior.
- If `benchmark_status == "passed"` and `perf_summary_source != "compare-perf"`, return `revise-required`.
- Keep artifact existence checks unchanged; this new check is specifically about the authority of the claimed performance conclusion.

## Testing

- Add prompt/guidance tests that pin the new `compare-perf` wording.
- Add round-gate tests showing that a benchmark-passing round is rejected when `perf_summary_source` is missing or set to any non-authoritative value.
- Update runtime fixtures so their fake passing rounds include `perf_summary_source: "compare-perf"`.

## Expected Outcome

- Optimize rounds can still choose hypotheses freely, but they must justify wins with the same comparison tool every time.
- The CLI no longer accepts a round that claims a benchmark win based on agent-invented math.
