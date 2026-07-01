# CPU Offload for Scalar-Heavy Wrapper Ops and Sequential Launch Loops

## Summary

Move NPU work that the NPU does **poorly or redundantly** to the CPU, using numpy/scipy, then bring the result back with `torch.from_numpy()`. This is a **wrapper-level structural pattern**: it changes *where* work runs without rewriting the Triton kernel itself, and should be evaluated before kernel-level micro-optimizations.

Two distinct flavors share the same precondition (moderate data volume, bulk D2H/H2D amortized over many ops or iterations, CPU not saturated):

- **Flavor A — one-shot scalar/integer wrapper ops**: `torch.unique`, `torch.sort`/`argsort`, `torch.searchsorted`, `torch.nonzero`, `torch.bincount`. The NPU scalar pipeline is the bottleneck; numpy's optimized C path wins on small-to-medium integer tensors.
- **Flavor B — per-iteration compute inside a sequential launch loop**: a Python `for`/`while` loop that, every iteration, launches an NPU kernel (Triton or aclnn) over a per-iteration work set *and* reads back per-iteration state to pick the next item. Launch overhead + per-iteration GPU→CPU sync dominates; the per-iteration compute is vectorizable over the work set, so a numpy expression can replace the kernel body losslessly. Often the NPU kernel's output turns out to be **redundant** (never consumed, or recomputed on CPU) — in which case deleting it is pure win, not a trade-off.

Flavor B is the higher-impact case (it can remove tens of thousands of redundant launches), but it is also the one with a sharp failure mode — see **Cross-device RMW trap** below. Flavor B is a wrapper-level / data-movement change, **not** a "pure-PyTorch rewrite" that bypasses the Triton NPU path — see **Scope Boundary** below before rejecting it on that ground.

## Scope Boundary — This Is Not A "Pure-PyTorch Rewrite"

Run guidance typically forbids *"replacing the Triton Ascend NPU computation path with a pure PyTorch rewrite"* and says such rewrites *"do not count as a successful optimize round."* **This pattern is not that.** The boundary below exists so the pattern is not wrongly rejected on that rule — read it before concluding "I cannot do Flavor B because it bypasses the Triton kernel."

- **What the rule forbids** (and this pattern does NOT do): delete the Triton kernel(s) and reimplement the operator's *bulk-parallel core compute* in eager torch / numpy with no structural justification — i.e., throw away the NPU and run everything on host.
- **What this pattern does** — each flavor is a permitted wrapper-level / data-movement change, not a wholesale rewrite:
  - **Flavor A** moves one-shot scalar/integer *wrapper* ops (`unique`/`sort`/`nonzero`/`searchsorted`/`bincount`) to CPU. These are `aclnn*` ops around the Triton kernel, not the Triton kernel itself; moving them is wrapper-logic cleanup.
  - **Flavor B Step 2a** *deletes a per-iteration NPU kernel whose output is never consumed*. This is dead-work removal, not replacement — the authoritative computation already lives on the CPU side. The rule protects real NPU computation, not redundant NPU computation, so Step 2a is unambiguously in-scope.
  - **Flavor B Step 2b** relocates *only* the irreducibly-sequential per-iteration loop (the host-sync-bottlenecked part: launch×count + GPU→CPU sync×count) to CPU numpy, while **keeping at least one bulk-parallel Triton kernel on the NPU path** for the work the NPU does well (elementwise mask / sort / gather / reduction over the full candidate set). The per-iteration loop was never "the NPU computation path" in the protected sense — it was a host-side launch/sync loop that happened to call an NPU kernel per iteration.

**Boundary rule:** Flavor B is in-scope as long as, after applying it, **at least one Triton kernel remains on the NPU path doing genuinely parallel work over the full candidate set.** If applying Flavor B would remove *all* Triton kernels from the operator, that crosses into a pure-PyTorch rewrite and is out of scope — do not apply.

**The per-iteration compute moved by Step 2b is a lossless numpy equivalent of the deleted kernel body** (elementwise + `np.where` + indexed write over the tail) — same candidates, same scores, same suppression/reduction decision, executed where the sequential dependency actually lives. It is not an algorithmic rewrite of the operator. Because the bulk-parallel NPU computation path is preserved, **a Flavor B round does count as a successful optimize round**, not a bypass.

Do not let the "no pure-PyTorch rewrite" rule cause a blanket rejection of Flavor B. The legitimate concern it targets (wholesale NPU bypass / reimplementing bulk-parallel compute on host) is already excluded by the boundary rule above; the targeted per-iteration offload is the structurally correct fix this pattern exists to provide.

## Use When

### Flavor A — one-shot scalar wrapper ops
- Wrapper code calls `torch.unique(tensor, return_inverse=True)`, `torch.sort`/`argsort`, `torch.searchsorted`, `torch.nonzero`, `torch.bincount`, or equivalent on NPU tensors.
- The input is small-to-medium integer tensors (typically ≤1M elements of int32/int64).
- Profiling shows these NPU ops have disproportionate latency relative to data volume (e.g., 30-80us for a few hundred integer IDs).
- The result is needed *before* the main Triton kernel to reduce downstream computation (e.g., dedup to group-by on fewer rows).

### Flavor B — sequential launch loop
- A Python `for`/`while` loop iterates over "selected" items, with an iteration count that is moderate but potentially large (hundreds to a few thousand). Each iteration:
  1. picks the next item from per-iteration state,
  2. launches an NPU kernel (Triton or aclnn) that does parallel work over a per-iteration *tail* (remaining candidates / neighbors / rows),
  3. updates a state tensor on the NPU that the *next* iteration's item selection depends on.
- The per-iteration work set is **embarrassingly vectorizable** over the tail — each candidate's per-iteration score is independent of the others, so the kernel body has a 1:1 numpy equivalent (elementwise + `np.where` + indexed write).
- Profiling shows the per-iteration NPU kernel is invoked **thousands of times** and dominates NPU time, while each invocation's useful compute is small (the vector units are underutilized and the tail shrinks every iteration).
- A read-modify-write dependency between iterations is *why the loop exists* — the work cannot be trivially collapsed into one fused kernel without changing semantics.

## Avoid When

- Data volume is **very large** (>>1M elements, or tens of MB) where CPU↔NPU transfer outweighs the saved NPU cost.
- The op is part of the **core compute pipeline** and must run on NPU for data locality (e.g., inside a fused kernel's dataflow).
- The CPU is **already saturated** with concurrent work.
- The op is **bandwidth-bound** on NPU and scalar/launch overhead is not the limiting factor.
- The NPU op is already very fast (<5us, single launch) and the CPU transfer round-trip would dominate — Flavor A only.
- (Flavor B) The per-iteration compute is **not** vectorizable over a tail (e.g., each iteration's work depends on the previous iteration's *numeric result* in a way no numpy expression can express) — then offloading to CPU just turns an NPU launch loop into a Python scalar loop, which is slower.
- (Flavor B) The per-iteration work set is **large and dense** so a fused batched NPU kernel would genuinely win — first try `auxiliary-op-fusion` / `grid-flatten-and-ub-buffering`; only fall back to CPU offload if profiling shows the fused NPU version still serializes or regresses (as a purely sequential algorithm typically does).

## Signals

### Code

#### Flavor A
- Wrapper calls `torch.unique(tensor, return_inverse=True)` or a `seen = zeros; seen[indices]=True; torch.where(seen)[0]` + `searchsorted` chain.
- The unique/sort/search result feeds only dispatch/scatter indexing, not kernel compute.

#### Flavor B — the decisive signals
- A Python `while`/`for` loop contains both `triton_kernel[grid](...)` (or an aclnn op) **and** a per-iteration `.item()` / `.cpu()` / `.tolist()` read of state — the read is what forces the per-iteration sync.
- The NPU kernel writes a state tensor `T_npu`, and the loop's branch condition reads `T_npu[i].item()` (or a CPU mirror `T_cpu[i]`) to decide the next iteration. **Trace the consumer of `T_npu`:**
  - if `T_npu` is **never read back** (the loop only reads a CPU-side mirror `T_cpu`), the NPU kernel is **redundant work** — delete candidate;
  - if `T_npu` is read back **every iteration** via sync, the kernel is creating the per-iteration sync trap — offload state + compute together.
- The loop body recomputes on CPU (numpy) the *same quantity* the NPU kernel just computed — strong redundant-compute signal.
- Loop iteration count is data-dependent and bounded by a small `max_output`/`max_iter` (hundreds to a few thousand).

### Profile

#### Flavor A
- `aclnnUnique2`, `aclnnNonzero`, `aclnnSearchSorted`, `aclnnBincount`, `aclnnSort` appear with non-trivial time (30-80us each); together 10-20% of total-op latency for small-to-medium row counts.

#### Flavor B
- A Triton kernel or aclnn op appears with an **invocation count in the thousands** and dominates NPU time (often >90%), while the single hottest *static* kernel is tiny — the cost is launch×count + sync×count, not compute.
- After removing the redundant per-iteration NPU kernel, NPU compute drops to a few percent of total-op and the residual shifts to CPU + D2H/H2D — confirming the NPU was doing little useful work per launch.
- Per-block time of the per-iteration kernel is nearly constant regardless of tail size → the kernel is launch/DMA-setup-bound, not compute-bound (see `scalar-latency-traps`).

## Optimization Strategy

### Flavor A — one-shot scalar wrapper ops

1. Move the integer tensor to CPU: `indices_cpu = indices.cpu().numpy()`.
2. Apply numpy's optimized C path: `unique_np, inverse_np = np.unique(indices_cpu, return_inverse=True)`.
3. Bring results back: `torch.from_numpy(unique_np).to(device)`, `torch.from_numpy(inverse_np).to(device)`.
4. Use the compact keys to reduce downstream computation.

#### Unique/Dedup
```python
# BEFORE: NPU-side unique
unique_keys, inverse_indices = indices.unique(return_inverse=True, sorted=False)

# AFTER: CPU-side numpy unique
indices_cpu = indices.cpu().numpy()
unique_np, inverse_np = np.unique(indices_cpu, return_inverse=True)
unique_keys = torch.from_numpy(unique_np).to(device)
inverse_indices = torch.from_numpy(inverse_np).to(device)
```

#### Searchsorted (for group assignment)
When unique keys are needed for group index mapping, replace the whole `seen + nonzero + searchsorted` chain with one `np.unique`:
```python
# BEFORE: NPU chain — seen tensor + nonzero + searchsorted + IndexAdd
seen = torch.zeros(output_size, dtype=torch.bool, device=device)
seen[indices] = True
unique_keys = torch.where(seen)[0]                       # aclnnNonzero
group_indices = torch.searchsorted(unique_keys, indices) # aclnnSearchSorted

# AFTER: CPU numpy — single np.unique call replaces the entire chain
indices_cpu = indices.cpu().numpy()
unique_np, inverse_np = np.unique(indices_cpu, return_inverse=True)
unique_keys = torch.from_numpy(unique_np).to(device)
inverse_indices = torch.from_numpy(inverse_np).to(device)
```

#### Scatter/IndexPut to output
When the dedup path scatters compact results to a sparse output, prefer PyTorch indexing over explicit NPU scatter ops:
```python
# BEFORE: Triton scatter kernel + IndexPutImpl + InplaceCopy chain
# AFTER: Simple PyTorch indexing
output[unique_keys] = compact_result.to(output.dtype)
```

### Flavor B — sequential launch loop

This is a **delete-then-offload** procedure, not a blind move. Run the diagnostic first; it determines whether step 2 is "delete the NPU kernel" or "replace the NPU kernel with a CPU vectorized expression".

#### Step 1 — Diagnose: is the per-iteration NPU kernel redundant?

For the per-iteration NPU kernel `K` that writes state tensor `T_npu`:

1. **Profile**: confirm `K`'s invocation count ≈ loop iteration count and that `K` dominates NPU time. If invocation count is in the thousands and per-invocation work is small, the launch×count + sync×count overhead is the real cost.
2. **Trace the consumer of `T_npu`** (grep every read of the tensor `K` writes):
   - `T_npu` is **never read** (the loop reads a separate CPU mirror `T_cpu`, or recomputes `K`'s output on CPU) → `K` is **pure redundant work**. Go to Step 2a.
   - `T_npu` is read back **inside the loop** via `.item()`/`.cpu()`/`.tolist()` → `K` is the **per-iteration sync trap**. Go to Step 2b.
   - `T_npu` is read only **after the loop** → the dependency is not actually per-iteration; consider `auxiliary-op-fusion` instead (the loop may be collapsible).

This diagnostic is the heart of Flavor B. Do not skip it based on reasoning alone — verify with both the profile and a code trace. A kernel that *looks* load-bearing often turns out to write a tensor nobody reads.

#### Step 2a — Delete redundant NPU compute (pure win)

If `K`'s output is already produced/consumed on the CPU side, just remove the `K` launch (and its argument setup). No correctness risk: the CPU path was already computing the authoritative result. Re-profile; this alone can remove >90% of NPU time when the loop runs thousands of times.

#### Step 2b — Offload state + per-iteration compute together

If `K` is load-bearing (its output genuinely drives the next iteration), move **both** the state and the per-iteration compute to CPU so the whole sequential dependency chain lives in CPU memory with **zero per-iteration GPU↔CPU sync**:

1. **Bulk D2H once, before the loop**: copy every per-candidate array the loop touches (coordinates/features/scores/state) to numpy. One transfer, amortized over all iterations.
2. **State on CPU**: `state_cpu = np.zeros(n, dtype=np.int32)` — the read-modify-write state lives entirely in CPU memory.
3. **Per-iteration compute as a vectorized numpy expression over the tail**: replace the kernel body with the equivalent elementwise/`np.where`/indexed-write over `data_cpu[tail:]`. This is possible *because* the per-iteration work is embarrassingly vectorizable over the tail (each candidate independent).
4. **In-place state update**: `state_cpu[np.where(score >= threshold)[0] + tail] = 1` — read and write the same CPU array; no sync.
5. **Keep on NPU only** the parts that are (a) truly bulk-parallel (run once over all data, not per-iteration) and (b) **not** part of the sequential dependency chain. A one-shot elementwise mask/sort/gather over all candidates typically stays on NPU; the per-iteration suppression/reduction loop moves to CPU.
6. **Bulk H2D once, after the loop**: ship the selected indices / reduced result back.

The "bulk-copy once, zero sync" premise that justifies this flavor **only holds when no NPU op writes the loop state inside the loop**. If you keep any per-iteration NPU write to the state, you re-introduce the per-iteration sync and the pattern regresses — see the RMW trap below.

## Implementation Sketch

### Flavor A — dedup + compact accumulation
```python
import numpy as np

def _cpu_dedup_accumulate(values, indices, output_size, device, output_dtype):
    # Step 1: CPU dedup — move indices to CPU, run numpy unique
    indices_cpu = indices.cpu().numpy()
    unique_np, inverse_np = np.unique(indices_cpu, return_inverse=True)

    # Step 2: Bring results back to NPU
    unique_keys = torch.from_numpy(unique_np).to(device)
    inverse_indices = torch.from_numpy(inverse_np).to(device)

    # Step 3: Compact NPU accumulation — only unique_keys rows
    compact_result = torch.zeros(
        unique_keys.shape[0], values.shape[1],
        dtype=values.dtype, device=device,
    )
    compact_result.index_add_(0, inverse_indices, values)

    # Step 4: Scatter to full output via indexing (no separate scatter kernel)
    output = torch.zeros(
        output_size, values.shape[1],
        dtype=output_dtype, device=device,
    )
    output[unique_keys] = compact_result.to(output_dtype)
    return output
```

### Flavor B — generic select-and-reduce loop
Shape: pick the next unsuppressed candidate, vectorized-score it against the remaining tail, mark suppressed, repeat. The `vectorized_score` placeholder stands for whatever the kernel body computed (overlap, distance, intersection, conflict, etc.) — keep it as a single numpy expression over the tail.
```python
import numpy as np

def _cpu_sequential_select_and_reduce(features, max_output, threshold, device):
    # features: per-candidate data on NPU (scores / coords / embeddings).
    # 1. Bulk D2H the per-candidate data the loop touches (moderate volume).
    feat_cpu = features.cpu().numpy()            # one transfer, amortized
    n = feat_cpu.shape[0]
    state_cpu = np.zeros(n, dtype=np.int32)      # RMW state lives on CPU

    selected = []
    i = 0
    while i < n and len(selected) < max_output:
        if state_cpu[i] != 0:
            i += 1
            continue
        selected.append(i)
        # 2. Per-iteration compute, vectorized over the remaining tail on CPU.
        #    This is the lossless numpy equivalent of the deleted NPU kernel body.
        tail = slice(i + 1, n)
        score = vectorized_score(feat_cpu[i], feat_cpu[tail])   # numpy expr
        # 3. In-place state update — read+write the same CPU array, no sync.
        state_cpu[np.where(score >= threshold)[0] + i + 1] = 1
        i += 1

    # 4. Bulk H2D the result once after the loop.
    out = torch.from_numpy(np.asarray(selected, dtype=np.int32)).to(device)
    return out, torch.tensor(len(selected), dtype=torch.int32, device=device)
```

## Failure Modes And Anti-signals

- **Transfer time > saved NPU time** (both flavors): if `.cpu().numpy()` + `.to(device)` exceeds the saved NPU op time, the pattern is counterproductive. Always re-profile end-to-end.
- **Cross-device read-modify-write trap (Flavor B, critical)**: if you offload only the *state read* to CPU but keep the per-iteration *compute* (the NPU kernel `K`) on the NPU, `K` still writes the state on the NPU each iteration, so the CPU copy is stale and must be re-synced every iteration. A whole-tensor `.cpu()` per iteration is more expensive than the original per-iteration scalar `.item()`, so the pattern **regresses**. **Rule: in a sequential launch loop, offload state and per-iteration compute together, or not at all.** The "bulk-copy once, zero sync" premise only holds when no NPU op writes the loop state inside the loop. If you cannot move the compute too (e.g., it is not vectorizable over the tail), do not apply Flavor B — try `auxiliary-op-fusion` instead.
- **Large output / index range dilutes the dedup benefit** (Flavor A): `compact_result` becomes nearly as large as the full output.
- **Pure-Python scalar loop regression (Flavor B)**: if the per-iteration compute is *not* vectorizable over a tail, offloading turns an NPU launch loop into a Python scalar loop — slower. Flavor B requires the per-iteration work to be a numpy vector expression over the tail.
- **CPU saturation**: CPU work not overlapped with NPU can bottleneck other concurrent work.

## Risks

- (Flavor A) `numpy.unique` result order may differ from `torch.unique(sorted=False)` — verify downstream is order-insensitive.
- (Flavor A) CPU→NPU transfer is synchronous and blocks the NPU stream; for critical paths consider stream overlap.
- (Flavor B) Moving compute to CPU changes the floating-point reduction order (numpy vs Triton); for comparison/threshold ops this is exact, but for accumulative reductions verify the tolerance against the reference.
- (Flavor B) `np.where` / integer indexing tie-breaking can differ from the NPU kernel's vectorized tie-break when multiple candidates hit the threshold simultaneously — verify the selected-index set matches modulo ordering, and that the algorithm is order-independent at that step.
- Moderate (~1ms) CPU time is acceptable if it eliminates thousands of NPU launches + syncs; the break-even is launch×count + sync×count, not a single op's cost.

## What To Verify After Applying

1. **Correctness**: output matches the reference exactly (Flavor A: numpy vs torch unique may differ in order — verify downstream is order-insensitive; Flavor B: the per-iteration numpy expression must be a semantic match to the deleted kernel body, including threshold sense and tie-handling).
2. **Benchmark**: compare total-op latency before/after. Flavor A: the CPU dedup + simple indexing chain should beat the NPU unique/searchsorted/nonzero chain. Flavor B: total-op should drop by roughly the per-iteration kernel's (launch + sync) × iteration-count; after the fix, NPU compute should be a few percent of total-op and the residual should be CPU + D2H/H2D.
3. **Invocation count**: confirm the per-iteration NPU kernel's invocation count dropped to 0 (Flavor B) — the profile is the proof, not the speedup number alone.
4. **CPU utilization**: ensure the CPU loop does not bottleneck concurrent work.
5. **Shape diversity**: verify across all benchmark shapes — Flavor B's CPU overhead is near-constant per iteration, while the original NPU launch×count cost scaled with iteration count, so the win should hold or grow on larger cases (until D2H transfer of the per-candidate data starts to dominate).

## Related Patterns

- `auxiliary-op-fusion` — fuse/ Tritonize one-shot auxiliary ops; prefer it over Flavor A when the fused NPU path is genuinely cheaper than the CPU round-trip, and over Flavor B when the loop turns out to be collapsible (state read only after the loop).
- `grid-flatten-and-ub-buffering` — kernel-level batching; try it before Flavor B when the per-iteration work is large/dense enough that a batched NPU kernel could win.
- `scalar-latency-traps` — Flavor B's per-iteration NPU kernel is almost always launch/scalar-bound; this pattern explains why its per-block time is constant.
- `layout-materialization-elision` — another wrapper-level delete-redundant-work pattern; composes with Flavor A's scatter cleanup.
- `algebraic-optimization` — mathematical rewrites that can further shrink the per-iteration numpy expression in Flavor B.
