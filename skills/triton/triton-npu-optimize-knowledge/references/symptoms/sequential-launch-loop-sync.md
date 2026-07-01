# sequential-launch-loop-sync

## Summary

A Python-level `for`/`while` loop drives an NPU kernel (Triton or aclnn) **per iteration** over a shrinking per-iteration work set, and each iteration also reads per-iteration state back to the host (`.item()` / `.cpu()` / `.tolist()`) to decide the next item. The cost is **launch×count + GPU→CPU sync×count**, not useful compute; the NPU vector/cube units are underutilized on every launch. This is a host-side structural symptom, distinct from in-kernel scalar overhead (`high-scalar-overhead`).

## Evidence To Confirm

- Code inspection shows a Python `for`/`while` loop body that **both** launches an NPU kernel (Triton `kernel[grid](...)` or an `aclnn*` op) **and** reads a scalar / small tensor back via `.item()` / `.cpu()` / `.tolist()` in the same iteration.
- The read-back value (a per-iteration state tensor or scalar) gates the next iteration's item selection — a read-modify-write dependency *between* iterations that prevents collapsing the loop into one fused kernel.
- Profiling shows the per-iteration NPU kernel with an **invocation count in the thousands** (≈ loop trip count) that dominates NPU time, while each single invocation's useful compute is small and its per-block time is nearly constant regardless of tail size (launch/DMA-setup-bound).
- The per-iteration work is **embarrassingly vectorizable over a tail** of remaining candidates (each candidate's per-iteration score is independent of the others).
- **Redundant-compute signal:** trace the tensor the per-iteration kernel writes — if the loop instead reads a separate CPU mirror or recomputes the same quantity on CPU, the NPU kernel's output is never consumed and the kernel is pure wasted work.

## Candidate Pattern Directions

- `cpu-offload-dedup` — Flavor B: delete redundant per-iteration NPU compute, or offload state + per-iteration compute **together** to CPU numpy so the whole loop runs with zero per-iteration GPU↔CPU sync. **Open the full pattern file and run its Step 1 diagnostic before dismissing.**
- `auxiliary-op-fusion` — only if the per-iteration kernel's output is read *after* the loop (the dependency is not actually per-iteration), in which case the loop may be collapsible into one fused NPU kernel.
- `grid-flatten-and-ub-buffering` — only if the per-iteration work set is large/dense enough that a batched NPU kernel could genuinely win; a purely sequential algorithm usually regresses here, so fall back to `cpu-offload-dedup`.

## Common Non-Matches

- A Python loop that launches an NPU kernel but does **no per-iteration host read-back** (no `.item()`/`.cpu()`/`.tolist()` gating the next iteration) is not this symptom — the loop is likely collapsible; prefer `auxiliary-op-fusion`.
- If the per-iteration compute is **not** vectorizable over a tail (each iteration depends on the previous iteration's *numeric result* in a way no numpy expression can express), CPU offload just turns an NPU launch loop into a slower Python scalar loop — do not apply `cpu-offload-dedup` Flavor B.
- Very large per-iteration work sets (>>1M elements or tens of MB) where D2H transfer outweighs saved launch/sync cost — the bulk-copy-once premise breaks.
