# Round Check Triton Kernel Continuity Design

## Summary

- Add a new round-check-only static validation that rejects optimize rounds which replace the Triton kernel execution path with pure PyTorch computation.
- Keep the check lightweight and file-local: inspect the round-local operator artifact and look for recognizable Triton kernel launch signals.
- Allow mixed PyTorch plus Triton operators as long as the round artifact still preserves a real Triton kernel launch path.
- Implement the detection in a dedicated Python module under `skills/triton-npu-optimize-check/scripts/`, then call it from the existing `check_round()` flow.

## Problem

- The current optimize prompts already tell the code agent not to bypass the Triton kernel path.
- The current `triton-npu-optimize-check` round gate does not enforce that policy. It checks artifact completeness and a small set of round-state semantics, but it does not inspect whether the operator still launches Triton.
- That leaves an obvious cheating path: a round can pass correctness and benchmark steps after rewriting the operator to call a PyTorch op directly, even though no Triton kernel optimization actually happened.

## Goals

- Enforce in `check_round()` that the optimized operator still contains a recognizable Triton kernel launch path.
- Permit valid mixed implementations where PyTorch handles wrapping, dispatch, or edge cases while Triton still performs the core kernel work.
- Keep the logic isolated in a new helper module with one clear responsibility.
- Return `revise-required` rather than `hard-fail` when the kernel continuity check fails so the round can be repaired in place.

## Non-Goals

- Do not require that the round eliminate all PyTorch operators.
- Do not introduce new baseline or round-state metadata fields.
- Do not attempt full semantic proof that every code path reaches Triton at runtime.
- Do not add profiler-based enforcement in this iteration.

## Recommended Approach

Create a new static analysis helper in `skills/triton-npu-optimize-check/scripts/` that:

- reads the round-local operator file discovered by the existing round artifact inspection
- detects whether the file still contains recognizable Triton kernel continuity signals
- returns a small structured result that the existing `check_round()` function can turn into gate issues

Then update `check_round()` so that:

- existing artifact and round-state checks still run first
- the new kernel continuity helper runs only after the round artifact exists and can be inspected
- missing Triton continuity becomes a `revise-required` round result with a concise actionable issue message

## Detection Policy

The first version should use conservative heuristics instead of trying to prove full call-graph reachability.

### Pass When

- the round-local operator still contains recognizable Triton kernel launch signals
- PyTorch wrapper logic is present, but Triton launch behavior is still clearly preserved

### Fail When

- the round-local operator shows only pure PyTorch-style computation signals for the main implementation
- no recognizable Triton kernel launch signal remains in the file

### Heuristic Signals

Use signals that are common in Triton operator source files, such as:

- Triton imports such as `import triton` or `import triton.language as tl`
- Triton kernel definitions such as `@triton.jit`
- Triton launch syntax such as `kernel[...](`, including common multiline formatting around the bracketed launch grid and call arguments

The important rule is not "PyTorch exists" but "Triton launch continuity still exists." A file that contains both should pass.

## Module Boundary

Add a dedicated helper module rather than growing `optimize_check_contract.py` further.

Suggested shape:

- one small dataclass for the analysis result
- one public function that accepts the round operator file path and returns whether Triton continuity was detected plus a short reason

`optimize_check_contract.py` should remain the orchestration layer that:

- discovers the round operator artifact
- invokes the helper
- converts helper failures into round issues

## Error Handling

- If round artifact inspection cannot find an operator file, keep the existing `missing round-local operator output` issue.
- If the helper can read the file but cannot find any recognizable Triton continuity signal, report a revise-required issue such as `round operator no longer preserves a recognizable Triton kernel launch path`.
- Avoid overly specific failure text that claims certainty about runtime behavior when the implementation is heuristic.

## Testing

Add narrow tests around the new behavior:

- a round-local operator that mixes PyTorch wrapper code with a recognizable Triton launch path should pass
- a round-local operator rewritten to pure PyTorch without Triton launch should return `revise-required`
- existing round artifact failure behavior should remain unchanged when the operator file is missing

Keep the tests focused on round-check behavior, not on the exact internal implementation of the helper.

## Expected Outcome

- Optimize rounds that silently swap the Triton kernel path for direct PyTorch computation will stop passing round check.
- Mixed PyTorch plus Triton operators will remain valid.
- The enforcement lives in one isolated helper module that can be strengthened later without tangling the rest of the round contract code.
