# Performance Question Playbook

## From Perf Symptom To Compiler Question

Rewrite the current round symptom into one narrow compiler-side question before reading source. Good starting forms are:

- which pass family would explain this stage transition?
- which subsystem would plausibly introduce this copy or sync behavior?
- which compiler-side constraint does this IR symptom imply for the current operator?

## Suspicious Stage Transition

- Start from the suspicious adjacent stages already identified in `opt-round-N/ir/`.
- Read pass docs first.
- Then inspect the matching `<compiler-source-dir>/bishengir/lib/` subtree.

## Vectorization Loss

- Confirm the loss in IR first.
- Read pass or feature docs for the likely vectorization subsystem.
- Inspect `<compiler-source-dir>/bishengir/lib/Conversion/` and `<compiler-source-dir>/bishengir/lib/Transforms/`.

## Copy Or Sync Growth

- Confirm the growth in IR or `perf-analysis.md`.
- Read feature docs first.
- Inspect dialect or conversion code under `<compiler-source-dir>/bishengir/lib/` that can introduce those operations.

## Buffer Expansion Or Memory-Planning Issue

- Confirm the symptom in IR first.
- Read feature docs for memory- or layout-related behavior.
- Inspect the matching implementation subtree under `<compiler-source-dir>/bishengir/lib/`.

## Fusion Or Lowering Shape Regression

- Confirm where the structure changes across stages.
- Read pass docs first, then inspect the implementation subtree under `<compiler-source-dir>/bishengir/lib/`.

## Turning Source Findings Into Operator Actions

Always finish by writing:

- the likely compiler-side explanation
- what that implies for the current Triton operator
- what the next operator change should target
