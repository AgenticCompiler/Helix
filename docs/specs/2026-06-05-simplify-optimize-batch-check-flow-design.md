# Simplify Optimize Batch Check Flow Design

## Summary

- Collapse optimize round check outcomes to `status: "pass" | "fail"`.
- Remove runtime gate-decision layering from the checked and supervised batch loop.
- Replace `_determine_batch_followup` with `check_batch_round`, which produces one merged previous-batch check result.
- Use that merged check result directly in the next worker prompt when the previous batch needs fixes.

## User-Visible Behavior

- `check-baseline` and `check-round` now expose `status` instead of `ok` and `decision`.
- `pass` means the checked artifacts are acceptable.
- `fail` means the checked artifacts need repair or are otherwise invalid.
- After each worker batch, the CLI checks every expected round directory from that batch and emits one merged result such as:

```text
opt-round-1: {"status":"pass",...}
opt-round-2: {"status":"fail",...}
Supervisor guidance: ...
```

- In `supervised` mode, the CLI runs one supervisor pass after the per-round checks and appends its guidance to the merged batch check result.
- If the merged previous-batch result contains any failure, the next worker prompt must say that it should first repair the previous batch issues before opening new rounds.
- If the merged previous-batch result contains no failures, the next worker prompt continues with the normal batch instructions.

## Runtime Design

### Check Result Model

- `OptimizeCheckResult` keeps:
  - `kind`
  - `status`
  - `issues`
  - `summary`
  - `next_option`
- The runtime normalizer should accept legacy skill-side `decision` and `ok` fields only as a compatibility input and map them to the new `status`.

### Batch Follow-Up

- `run_round_loop` owns one `followup_summary` string from the previous batch.
- For each batch:
  - build the worker request
  - run the worker
  - call `check_batch_round`
  - if the check result contains failures and another batch remains, pass that merged summary into the next worker prompt
  - if the check result contains failures and this was the final scheduled batch, stop the optimize run with that merged summary
  - otherwise advance to the next batch bounds
- The runtime does not rewind the next batch back to a `first_unresolved_round`, and it does not do a dedicated repair rerun for the same batch.

### `check_batch_round`

- Iterate `opt-round-N` directories from `batch_start` through `batch_end`.
- Run `check_round` for each existing expected round directory.
- If a round directory is missing, record it as a failed entry.
- Build one merged text block with one line per round result.
- When `round_mode == "supervised"`, run one supervisor agent for the batch and append its report summary as `Supervisor guidance: ...`.
- Return:
  - the merged text block
  - whether any batch issue exists
- Do not short-circuit later round checks after the first failed round. Every expected round in the batch is checked and included in the merged result.

## Prompt Design

- Worker batch prompts still state:
  - `This invocation owns rounds X through Y.`
  - `Execute those rounds strictly one at a time.`
  - `Do not pre-plan the full batch before acting.`
- When previous-batch failures exist, prepend guidance like:
  - `This invocation needs to complete rounds X through Y, but before that, fix the previous batch issues.`
  - `Issues:`
  - the merged batch check result

## Cleanup

- Delete obsolete gate-oriented runtime helpers and types after the new flow lands.
- Remove old code paths that only exist to distinguish `revise-required`, `revise-metadata`, and `hard-fail`.
