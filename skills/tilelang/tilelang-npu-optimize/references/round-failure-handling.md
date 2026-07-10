# Optimize Round Failure Handling

## Typical Triggers

- An optimized round fails `run-test`
- An optimized round passes correctness but fails `run-bench`
- An optimized round passes both tools but shows a performance regression or no meaningful gain

## Expected Inputs

- the round operator file under `opt-round-N/`
- selected test mode and benchmark mode
- stderr, traceback, assertion output, comparison output, or profiler output
- the parent round identifier
- the round hypothesis or optimization theme

## Handling Priorities

1. Preserve the round operator path.
2. Repair the optimized operator itself, not the generated test or benchmark harness, unless the evidence clearly shows the harness is wrong.
3. Keep the optimization intent whenever possible.
4. Re-run correctness before trusting any follow-up benchmark.

## Correctness Failure Rules

- Start from the optimized round file, not from the original operator.
- Infer the most likely failure type from the raw evidence before editing.
- Typical categories:
  - shape or indexing mistakes
  - mask misuse
  - dtype or cast mistakes
  - load/store ordering bugs
  - reduction semantics changes
  - environment or import failures
- Prefer minimal fixes that keep the performance-oriented structure intact.

## Benchmark Failure Rules

- Distinguish harness failure from operator failure.
- If the benchmark harness is broken, repair only what is necessary to recover a valid measurement.
- If the operator regressed, revise the operator code first.
- Preserve the evidence that explains why the round did not help.

## Regression Handling

If a round is correct but slower:

- decide whether the round idea is still promising
- if yes, keep iterating within the same round
- if not, preserve the round summary as a failed or non-promoted branch and choose a different validated parent for the next round

Do not promote a slower round to current best status.

## Retry Expectation

After every repair:

1. run `run-test --verbose`
2. only if correctness passes, run `run-bench`
3. update the round summary with what was tried and what changed

## Preservation Rules

- Never overwrite the original operator file.
- Never silently drop the round hypothesis from the notes.
- Preserve comments that explain optimization intent.
