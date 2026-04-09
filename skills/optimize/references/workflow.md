# Optimize Workflow

## Goal

Turn operator optimization into a repeatable search process over validated candidates.

## Core Model

- Treat the original operator as `round 0`.
- Maintain a pool of validated candidates.
- A validated candidate is a version that passes correctness validation.
- The current best candidate is useful for reporting, but it is not the only legal parent for future rounds.
- New rounds may branch from any validated candidate when that branch better preserves search diversity or explores a different optimization idea.

## Pre-Round Setup

1. Resolve the operator workspace.
2. Resolve the correctness mode for this optimize run, defaulting to `differential` unless the task explicitly says otherwise.
3. Resolve the benchmark mode for this optimize run, defaulting to `standalone` unless the task explicitly says otherwise.
4. Check whether a correctness test for the resolved mode already exists and reuse it when possible.
5. Check whether a benchmark case for the resolved mode already exists and reuse it when possible.
6. Generate only the missing validation artifact types.
6. If the original operator has never been benchmarked in the current workspace, run a baseline benchmark before optimizing so later rounds have a stable comparison point.
7. Initialize `opt-note.md` if it does not exist.
8. Write a short diagnosis summary before the first code-changing round so later readers can see the suspected bottleneck and the initial evidence.

## Candidate Selection

Choose a parent from validated candidates, not only from the current best.

Prefer parents that satisfy one of these conditions:

- They are the current best performer.
- They preserve correctness and show a different optimization direction from the best performer.
- Their summary suggests an unfinished follow-up idea with clear next steps.
- They improve one bottleneck while leaving another bottleneck open for a second-stage optimization.

Avoid selecting a parent that:

- has unresolved correctness trouble
- lacks benchmark evidence
- already failed repeatedly under the same optimization idea

## Round Lifecycle

1. Allocate the next `opt-round-N/` directory.
2. Copy the chosen parent operator into the new round directory.
3. Keep the copied operator filename stable enough that `run-test` and `run-bench` can operate on it directly.
4. Create `opt-round-N/attempts.md` immediately and use it as the running log for the round.
5. State the round hypothesis before editing, for example:
   - better tiling
   - more parallel load order
   - reduced unnecessary masking
   - software pipelining
6. Record the initial hypothesis in `attempts.md`, including why it may help and what evidence supports it.
7. If you are not collecting profiling or IR evidence for this round, record why the current code or benchmark evidence is already sufficient.
8. Apply the optimization.
9. Record the code change in `attempts.md`.
10. Run correctness validation with the resolved correctness mode.
11. If correctness fails, record the failure in `attempts.md`, repair the operator in place, and retry.
12. After correctness passes, run the benchmark with the resolved benchmark mode.
13. Record the benchmark result in `attempts.md`.
14. If profiling or IR capture is needed to explain or validate the round, archive that evidence under `opt-round-N/profile/` or `opt-round-N/ir/`.
    - For IR capture, prefer commands shaped like:
      ```bash
      python3 ../ascend-operator-ir-analyzer/scripts/capture_ir.py --ir-dir opt-round-N/ir --bench-file bench_<operator>.py --operator-file opt-round-N/<optimized-operator>.py
      python3 ../ascend-operator-ir-analyzer/scripts/inspect_ir.py list-stages --ir-dir opt-round-N/ir --sort-by interesting --limit 20
      ```
15. If the benchmark regresses, either:
   - revise the round in place if the optimization idea is still promising, or
   - stop advancing that round and return to candidate selection for a new branch
16. Complete the round only after the optimized candidate shows a measurable win over the chosen comparison target.
17. When the optimize session is pausing or ending, refresh the final `## Overall Summary` block in `opt-note.md` so the top-level note states the best round, overall outcome, any useful validated branches, and why that round was pursued.

## Comparison Target

Compare the new round against:

- its direct parent, to prove the local optimization helped
- the current best candidate, when the round is meant to compete for best overall status

If the new round beats its parent but not the current best, still keep it when:

- it opens a different optimization direction
- it has a reusable technique worth preserving
- it may compose well with a later round

## Completion Criteria

A round is complete only when all of the following are true:

- the optimized operator passes correctness validation
- `attempts.md` captures the meaningful intermediate trials within the round
- benchmark evidence is saved
- any profiler or IR evidence used in the round decision is saved under the round directory
- the round summary explains the optimization points and measured outcome
- `opt-note.md` is updated with a concise entry
- `opt-note.md` ends with one current `## Overall Summary` block for the whole session

## Failure Handling

- Use [round-failure-handling.md](round-failure-handling.md) when correctness or benchmark execution fails.
- Do not overwrite the original operator.
- Do not erase a useful failed attempt if its summary would help a future branch; keep the artifacts when they contain reusable learning.
