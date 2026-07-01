# CPU Offload for Scalar-Heavy Wrapper Ops, Sequential Launch Loops, and Small-Data Wrapper Helpers

## Summary

Move NPU work that the NPU does **poorly or redundantly** to the CPU, using numpy/scipy, then bring the result back with `torch.from_numpy()`. This is a **wrapper-level structural pattern**: it changes *where* work runs without rewriting the Triton kernel itself, and should be evaluated before kernel-level micro-optimizations.

Three distinct flavors share the same precondition (moderate data volume, bulk D2H/H2D amortized over many ops or iterations, CPU not saturated):

- **Flavor A — one-shot scalar/integer wrapper ops**: `torch.unique`, `torch.sort`/`argsort`, `torch.searchsorted`, `torch.nonzero`, `torch.bincount`. The NPU scalar pipeline is the bottleneck; numpy's optimized C path wins on small-to-medium integer tensors.
- **Flavor B — per-iteration compute inside a sequential launch loop**: a Python `for`/`while` loop that, every iteration, launches an NPU kernel (Triton or aclnn) over a per-iteration work set *and* reads back per-iteration state to pick the next item. Launch overhead + per-iteration GPU→CPU sync dominates; the per-iteration compute is vectorizable over the work set, so a numpy expression can replace the kernel body losslessly. Often the NPU kernel's output turns out to be **redundant** (never consumed, or recomputed on CPU) — in which case deleting it is pure win, not a trade-off.
- **Flavor C — one-shot small-data wrapper-helper pipeline**: a helper function (not the Triton kernel) builds small intermediate tensors (indices / weights / position ids / freq tables / cos·sin) through a chain of general-purpose NPU ops — `torch.arange`, `torch.linspace`, elementwise `mul/add/sub/clamp`, `torch.stack/cat`, `torch.cos/sin`, `torch.outer`, fancy-index gather, `torch.repeat`. No single op dominates, but their combined launch overhead does; each op's data is only a few KB–hundreds of KB. Rewriting the *entire helper* in numpy and shipping the result back with one `torch.from_numpy().to(device)` replaces N aclnn launches with one H2D copy.

Flavor B is the higher-impact case (it can remove tens of thousands of redundant launches), but it is also the one with a sharp failure mode — see **Cross-device RMW trap** below. Flavor C is the **highest-frequency case in practice** (most operators have small-data pre/post-processing helpers around their Triton kernel) and the one most easily missed: the ops look "essential" because their output feeds the kernel, but they are wrapper logic, not bulk-parallel compute — see the **"core compute" clarification** under Scope Boundary before concluding "I cannot move cos/sin/freqs because they are core compute." Flavor B and Flavor C are both wrapper-level / data-movement changes, **not** "pure-PyTorch rewrites" that bypass the Triton NPU path — see **Scope Boundary** below before rejecting either on that ground.

## Scope Boundary — This Is Not A "Pure-PyTorch Rewrite"

Run guidance typically forbids *"replacing the Triton Ascend NPU computation path with a pure PyTorch rewrite"* and says such rewrites *"do not count as a successful optimize round."* **This pattern is not that.** The boundary below exists so the pattern is not wrongly rejected on that rule — read it before concluding "I cannot do Flavor B/C because it bypasses the Triton kernel."

- **What the rule forbids** (and this pattern does NOT do): delete the Triton kernel(s) and reimplement the operator's *bulk-parallel core compute* in eager torch / numpy with no structural justification — i.e., throw away the NPU and run everything on host.
- **What this pattern does** — each flavor is a permitted wrapper-level / data-movement change, not a wholesale rewrite:
  - **Flavor A** moves one-shot scalar/integer *wrapper* ops (`unique`/`sort`/`nonzero`/`searchsorted`/`bincount`) to CPU. These are `aclnn*` ops around the Triton kernel, not the Triton kernel itself; moving them is wrapper-logic cleanup.
  - **Flavor B Step 2a** *deletes a per-iteration NPU kernel whose output is never consumed*. This is dead-work removal, not replacement — the authoritative computation already lives on the CPU side. The rule protects real NPU computation, not redundant NPU computation, so Step 2a is unambiguously in-scope.
  - **Flavor B Step 2b** relocates *only* the irreducibly-sequential per-iteration loop (the host-sync-bottlenecked part: launch×count + GPU→CPU sync×count) to CPU numpy, while **keeping at least one bulk-parallel Triton kernel on the NPU path** for the work the NPU does well (elementwise mask / sort / gather / reduction over the full candidate set). The per-iteration loop was never "the NPU computation path" in the protected sense — it was a host-side launch/sync loop that happened to call an NPU kernel per iteration.
  - **Flavor C** rewrites an entire small-data *wrapper helper* (the code that builds the kernel's inputs — indices/weights/pos_ids/freqs/cos·sin) in numpy, then ships the result back with one `.to(device)`. The helper is pre/post-processing around the Triton kernel, not the kernel itself; moving it is wrapper-logic cleanup, exactly like Flavor A. See the **"core compute" clarification** immediately below — it is the rule that stops Flavor C from being wrongly rejected as "moving core compute off the NPU."

**"Core compute" clarification (read before rejecting Flavor C):** the no-rewrite rule protects the **bulk-parallel Triton kernel doing genuinely parallel work over large data** — it does *not* protect small-data elementwise/construction wrapper ops that happen to produce the kernel's inputs. `torch.cos`/`torch.sin`/`torch.arange`/`torch.linspace`/`torch.cat` on a few-KB tensor are **wrapper ops, not core compute**, even when their output is consumed by the kernel. "Core compute" = bulk-parallel work over large data where NPU data locality wins (the gather/reduction/elementwise-over-millions-of-elements kernel); wrapper pre/post-processing on KB-scale data is Flavor C territory. A common failure mode is to see `cos`/`sin`/`freqs` feeding a RoPE/gather kernel, classify them as "core compute," and stop at 2–3x — when moving those exact wrapper helpers to CPU would break through to 15–30x.

**Boundary rule:** Flavor B and Flavor C are both in-scope as long as, after applying, **at least one Triton kernel remains on the NPU path doing genuinely parallel work over the full candidate set / the full output.** If applying Flavor B or Flavor C would remove *all* Triton kernels from the operator, that crosses into a pure-PyTorch rewrite and is out of scope — do not apply.

**The per-iteration compute moved by Step 2b is a lossless numpy equivalent of the deleted kernel body** (elementwise + `np.where` + indexed write over the tail) — same candidates, same scores, same suppression/reduction decision, executed where the sequential dependency actually lives. It is not an algorithmic rewrite of the operator. Because the bulk-parallel NPU computation path is preserved, **a Flavor B round does count as a successful optimize round**, not a bypass.

Do not let the "no pure-PyTorch rewrite" rule cause a blanket rejection of Flavor B or Flavor C. The legitimate concern it targets (wholesale NPU bypass / reimplementing bulk-parallel compute on host) is already excluded by the boundary rule above; the targeted per-iteration offload (B) and small-data wrapper-helper rewrite (C) are the structurally correct fixes this pattern exists to provide.

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

### Flavor C — one-shot small-data wrapper-helper pipeline
- A wrapper helper function (named `_build_*` / `_precompute_*` / `_make_*`) — **not** the Triton kernel — constructs small intermediate tensors via a chain of general-purpose NPU ops: `torch.arange`, `torch.linspace`, elementwise `mul/add/sub/clamp`, `torch.stack`/`torch.cat`, `torch.cos`/`torch.sin`, `torch.outer`, fancy-index gather (`freqs[pos_ids]`), `torch.repeat`.
- Each op's data is small (a few KB to a few hundred KB); the helper's total output is well under 1MB.
- No single aclnn op dominates, but there are many (often 8–20+ launches across the helper); their combined time is the majority of total-op.
- The Triton kernel itself is already cheap (typically <15% of total-op) — the bottleneck is wrapper-op launch overhead, not kernel compute. This is the signal that distinguishes Flavor C from kernel-level work: if the kernel is the bottleneck, this flavor does not apply.
- A `for` loop over grid/batch entries calls `.item()` per iteration to read NPU scalars (grid_thw rows, sizes), triggering a device→host sync each time.

## Avoid When

- Data volume is **very large** (>>1M elements, or tens of MB) where CPU↔NPU transfer outweighs the saved NPU cost.
- The op is **genuinely bulk-parallel compute over large data** that the NPU does well — i.e. the Triton kernel itself, or a large elementwise/reduction that is bandwidth-bound on NPU. Do **not** classify small-data elementwise/construction wrapper ops (`arange`/`cos`/`sin`/`cat`/`linspace` on a few-KB tensor) as "core compute" just because their output feeds the kernel — those are Flavor C candidates, not protected compute. "Core compute" = bulk-parallel work over large data where NPU data locality wins; wrapper pre/post-processing on small data is not that.
- The CPU is **already saturated** with concurrent work.
- The op is **bandwidth-bound** on NPU and scalar/launch overhead is not the limiting factor.
- The NPU op is already very fast (<5us, single launch) and the CPU transfer round-trip would dominate — Flavor A only.
- (Flavor B) The per-iteration compute is **not** vectorizable over a tail (e.g., each iteration's work depends on the previous iteration's *numeric result* in a way no numpy expression can express) — then offloading to CPU just turns an NPU launch loop into a Python scalar loop, which is slower.
- (Flavor B) The per-iteration work set is **large and dense** so a fused batched NPU kernel would genuinely win — first try `auxiliary-op-fusion` / `grid-flatten-and-ub-buffering`; only fall back to CPU offload if profiling shows the fused NPU version still serializes or regresses (as a purely sequential algorithm typically does).
- (Flavor C) The helper's per-op data is **large** (>>1MB) — the single H2D transfer would outweigh the saved launch overhead. Flavor C is for KB-scale helpers; if a helper produces MB-scale output, leave it on NPU or split it so only the small-data segment moves.
- (Flavor C) The Triton kernel is itself the bottleneck (>50% of total-op) — Flavor C only helps when the kernel is already cheap and the wrapper ops dominate. If the kernel is the bottleneck, go to kernel-level patterns instead.

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

#### Flavor C
- A helper function (named `_build_*` / `_precompute_*` / `_make_*`) is a sequence of `torch.arange`/`linspace`/`mul`/`add`/`clamp`/`stack`/`cat`/`cos`/`sin`/`outer`/fancy-index/`repeat` on NPU tensors, producing small intermediates that feed the Triton kernel (indices, weights, pos_ids, freqs, cos/sin tables).
- The helper body contains **no** bulk-parallel Triton kernel — it is pure wrapper logic.
- A `for` loop over grid/batch entries calls `.item()` per iteration to read NPU scalars (grid_thw rows, sizes) — each `.item()` is a device→host sync.
- The helper output is small (KB scale) but is built by 8–20+ separate `aclnn*` launches; the gather kernel that *consumes* the output is a single cheap Triton launch.

### Profile

#### Flavor A
- `aclnnUnique2`, `aclnnNonzero`, `aclnnSearchSorted`, `aclnnBincount`, `aclnnSort` appear with non-trivial time (30-80us each); together 10-20% of total-op latency for small-to-medium row counts.

#### Flavor B
- A Triton kernel or aclnn op appears with an **invocation count in the thousands** and dominates NPU time (often >90%), while the single hottest *static* kernel is tiny — the cost is launch×count + sync×count, not compute.
- After removing the redundant per-iteration NPU kernel, NPU compute drops to a few percent of total-op and the residual shifts to CPU + D2H/H2D — confirming the NPU was doing little useful work per launch.
- Per-block time of the per-iteration kernel is nearly constant regardless of tail size → the kernel is launch/DMA-setup-bound, not compute-bound (see `scalar-latency-traps`).

#### Flavor C
- Many distinct `aclnn*` ops appear (`aclnnArange`, `aclnnLinspace`, `aclnnMul`, `aclnnAdd`, `aclnnClamp`, `aclnnSub`, `aclnnStack`, `aclnnCat`, `aclnnCos`, `aclnnSin`, `aclnnIndex`, `aclnnRepeat`, `aclnnInplaceCopy`) each costing a few µs–tens of µs; **no single op dominates but their sum is >50% of total-op**. This "many small ops" profile is the signature — it is easy to look at each individually, conclude it is "essential," and stop, when the correct read is that the *collective* launch overhead is the bottleneck.
- The Triton gather/reduction kernel is only 3–15% of total-op — the residual is wrapper-op launch overhead, not compute. If the kernel were the bottleneck, this flavor does not apply.
- Per-op tensor sizes are in the KB range (a few hundred to a few hundred thousand elements); intermediate `expand().reshape()` / `view().permute().flatten()` chains trigger `aclnnInplaceCopy` to materialize non-contiguous tensors (10–50µs each).
- After rewriting the helper on CPU, NPU compute drops to just the Triton kernel (a few % of total-op) and the residual shifts to CPU + one H2D per helper — confirming the wrapper ops were the cost.

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

### Flavor C — one-shot small-data wrapper-helper pipeline

This is a **rewrite-the-whole-helper** procedure, not an op-by-op move. The win comes from replacing N launches + N contiguous-copies with one numpy computation + one H2D copy; doing it op-by-op forfeits most of the win because each individual op's savings barely covers its own transfer. The unit of offload is the *whole helper function*, not individual ops.

#### Step 1 — Identify the contiguous helper

Find the wrapper function whose body is a chain of small NPU ops on small data, whose output feeds the Triton kernel (indices / weights / pos_ids / freqs / cos·sin / permutation index). The whole function is the unit of offload. Typical names: `_build_*`, `_precompute_*`, `_make_*`. Confirm with the profile that this helper's ops collectively dominate total-op while the Triton kernel is <15% — if the kernel is the bottleneck, this flavor does not apply and you should go to kernel-level patterns.

#### Step 2 — Bulk D2H the inputs once, then rewrite the whole body in numpy

1. **Bulk D2H once at the top**: copy every small input the helper reads (grid metadata, inv_freq, etc.) to numpy: `grid_np = grid_thw.cpu().numpy()`. This also kills per-iteration `.item()` syncs — iterate over the numpy copy instead of calling `.item()` on NPU scalars.
2. **Rewrite the entire helper body in numpy** as one contiguous expression chain: `np.arange`/`np.linspace`/`np.multiply`/`np.add`/`np.clip`/`np.stack`/`np.concatenate`/`np.cos`/`np.sin`/`np.outer` + fancy indexing. No per-op `.to(device)` — keep everything in numpy until the end.
3. **Pin intermediate precision** to match torch: use `np.float32` (numpy defaults to float64, which diverges from torch's fp32 intermediates and breaks bf16/fp16 output matching — this is the #1 correctness trap). Do the final dtype conversion on NPU: `torch.from_numpy(result).to(device).to(dtype)`.
4. **Single bulk H2D at the end**: `torch.from_numpy(result_np.copy()).to(device)` — one transfer replaces N aclnn launches + their contiguous copies. Use `.copy()` when the numpy array is a view/non-contiguous.
5. **Keep on NPU**: the bulk-parallel Triton kernel, plus any genuinely large-data elementwise/reduction the NPU does well. The boundary: if a moved op's data is large (>~1MB) or bandwidth-bound, leave it on NPU.

#### Step 3 — Replace NPU spatial-permute chains with a CPU-precomputed index array

A common Flavor C sub-case: a `split → repeat → view → permute → flatten → cat` chain that merely reorders rows. Precompute the output-row-to-source-row mapping as an int index array on CPU (numpy), then replace the whole chain with one `torch.index_select(tensor, 0, perm_indices)` on NPU. This eliminates `aclnnRepeat` + `aclnnCat` + `aclnnInplaceCopy` (the materialization of non-contiguous permuted tensors) in exchange for one small int64 H2D transfer.

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

### Flavor C — rewrite a small-data wrapper helper on CPU
Concrete shape: a helper that builds interpolation indices/weights, or rotary-embedding cos/sin tables, for the kernel to consume. The whole helper is the unit of offload — one contiguous numpy chain, one H2D at the end.
```python
import numpy as np

def _build_interpolation_tables_cpu(grid_thw, num_grid_per_side, dtype, device):
    # 1. Bulk D2H once — also kills per-iteration .item() syncs.
    grid_np = grid_thw.cpu().numpy()
    index_chunks, weight_chunks, patch_sizes = [], [], []

    for _, h, w in grid_np:                       # iterate numpy, no .item()
        h_size, w_size = int(h), int(w)
        # 2. Entire helper body in numpy — one contiguous chain, no per-op .to(device).
        #    Pin np.float32 (numpy defaults to float64 → breaks bf16/fp16 matching).
        h_idxs = np.linspace(0, num_grid_per_side - 1, h_size, dtype=np.float32)
        w_idxs = np.linspace(0, num_grid_per_side - 1, w_size, dtype=np.float32)
        h_floor = h_idxs.astype(np.int64)
        w_floor = w_idxs.astype(np.int64)
        h_ceil = np.clip(h_floor + 1, 0, num_grid_per_side - 1)
        w_ceil = np.clip(w_floor + 1, 0, num_grid_per_side - 1)
        dh = h_idxs - h_floor.astype(np.float32)
        dw = w_idxs - w_floor.astype(np.float32)
        base_h, base_h_ceil = h_floor * num_grid_per_side, h_ceil * num_grid_per_side

        idx = np.stack([
            (base_h[:, None] + w_floor[None, :]).ravel(),
            (base_h[:, None] + w_ceil[None, :]).ravel(),
            (base_h_ceil[:, None] + w_floor[None, :]).ravel(),
            (base_h_ceil[:, None] + w_ceil[None, :]).ravel(),
        ], axis=0)
        wt = np.stack([
            ((1 - dh)[:, None] * (1 - dw)[None, :]).ravel(),
            ((1 - dh)[:, None] * dw[None, :]).ravel(),
            (dh[:, None] * (1 - dw)[None, :]).ravel(),
            (dh[:, None] * dw[None, :]).ravel(),
        ], axis=0)
        index_chunks.append(idx); weight_chunks.append(wt)
        patch_sizes.append(h_size * w_size)

    idx_np = np.concatenate(index_chunks, axis=1)
    weight_np = np.concatenate(weight_chunks, axis=1)
    # 3. Single H2D — one transfer replaces ~15 aclnn launches per grid entry.
    #    Final dtype conversion on NPU to match bf16/fp16 reference behavior.
    idx_tensor = torch.from_numpy(idx_np.copy()).to(device)
    weight_tensor = torch.from_numpy(weight_np.copy()).to(device).to(dtype)
    return idx_tensor, weight_tensor, patch_sizes
```
The same shape applies to rotary-embedding helpers: build `pos_ids`/`freqs`/`cos`/`sin` entirely in numpy (`np.outer`, fancy-index `freqs_np[pos_ids_np]`, `np.cos`/`np.sin`, `np.concatenate`), then ship the final cos/sin tensors back with one `.to(device)` each. The `np.cos`/`np.sin` on a few-hundred-KB tensor is **not** "core compute" — the bulk-parallel gather kernel that *consumes* them stays on NPU. A 4D-broadcasting coordinate builder (`block_rows[:,None,None,None]*N + intra_row[None,None,:,None]` → `expand` → `reshape` → `stack` → `repeat`) is the same case: replace it with `np.repeat`/`np.tile` flat construction on CPU and one H2D.

## Failure Modes And Anti-signals

- **Transfer time > saved NPU time** (all flavors): if `.cpu().numpy()` + `.to(device)` exceeds the saved NPU op time, the pattern is counterproductive. Always re-profile end-to-end.
- **Cross-device read-modify-write trap (Flavor B, critical)**: if you offload only the *state read* to CPU but keep the per-iteration *compute* (the NPU kernel `K`) on the NPU, `K` still writes the state on the NPU each iteration, so the CPU copy is stale and must be re-synced every iteration. A whole-tensor `.cpu()` per iteration is more expensive than the original per-iteration scalar `.item()`, so the pattern **regresses**. **Rule: in a sequential launch loop, offload state and per-iteration compute together, or not at all.** The "bulk-copy once, zero sync" premise only holds when no NPU op writes the loop state inside the loop. If you cannot move the compute too (e.g., it is not vectorizable over the tail), do not apply Flavor B — try `auxiliary-op-fusion` instead.
- **Large output / index range dilutes the dedup benefit** (Flavor A): `compact_result` becomes nearly as large as the full output.
- **Pure-Python scalar loop regression (Flavor B)**: if the per-iteration compute is *not* vectorizable over a tail, offloading turns an NPU launch loop into a Python scalar loop — slower. Flavor B requires the per-iteration work to be a numpy vector expression over the tail.
- **Precision drift (Flavor C, critical, common)**: numpy defaults to float64; torch intermediates are float32. If the helper feeds a bf16/fp16 kernel, float64 intermediates → `.to(dtype)` produce different rounding than torch's fp32-intermediate-then-cast path, failing the correctness check. **Always pin `np.float32`** on intermediate arrays and do the final dtype cast on NPU (`.to(dtype)`), matching the reference's intermediate precision. This is the #1 reason a Flavor C round fails correctness.
- **Op-by-op offload anti-pattern (Flavor C)**: moving one `torch.arange` to CPU while leaving the next `torch.mul` on NPU re-introduces a per-op H2D/D2H round-trip and usually regresses. The unit of offload is the *whole helper*, not individual ops — rewrite the entire function in numpy, then one H2D at the end.
- **Over-offloading / crossing the boundary (Flavor C)**: if the rewrite removes the *last* Triton kernel from the operator (e.g. you also numpy-ified the gather), that is a pure-PyTorch rewrite — out of scope. Keep at least one bulk-parallel Triton kernel on the NPU path.
- **CPU saturation**: CPU work not overlapped with NPU can bottleneck other concurrent work.

## Risks

- (Flavor A) `numpy.unique` result order may differ from `torch.unique(sorted=False)` — verify downstream is order-insensitive.
- (Flavor A) CPU→NPU transfer is synchronous and blocks the NPU stream; for critical paths consider stream overlap.
- (Flavor B) Moving compute to CPU changes the floating-point reduction order (numpy vs Triton); for comparison/threshold ops this is exact, but for accumulative reductions verify the tolerance against the reference.
- (Flavor B) `np.where` / integer indexing tie-breaking can differ from the NPU kernel's vectorized tie-break when multiple candidates hit the threshold simultaneously — verify the selected-index set matches modulo ordering, and that the algorithm is order-independent at that step.
- (Flavor C) numpy float64 vs torch float32 intermediates → bf16/fp16 output mismatch; pin `np.float32` and cast on NPU.
- (Flavor C) CPU loop over grid/batch entries is fine for a handful of entries (the typical case) but degrades if there are hundreds/thousands of entries with non-trivial per-entry work — then the loop itself becomes the bottleneck (consider vectorizing fully over all entries if the entry count is large).
- (Flavor C) `.copy()` before `torch.from_numpy` is required when the numpy array is non-contiguous or a view, else `from_numpy` errors or shares memory unexpectedly.
- Moderate (~1ms) CPU time is acceptable if it eliminates thousands of NPU launches + syncs; the break-even is launch×count + sync×count, not a single op's cost.

## What To Verify After Applying

1. **Correctness**: output matches the reference exactly (Flavor A: numpy vs torch unique may differ in order — verify downstream is order-insensitive; Flavor B: the per-iteration numpy expression must be a semantic match to the deleted kernel body, including threshold sense and tie-handling; Flavor C: the numpy rewrite must match the reference within dtype tolerance — for bf16/fp16 paths this is the critical check, verify intermediate precision is np.float32 and the final cast happens on NPU, do not trust a "passed" without checking the bf16 case specifically).
2. **Benchmark**: compare total-op latency before/after. Flavor A: the CPU dedup + simple indexing chain should beat the NPU unique/searchsorted/nonzero chain. Flavor B: total-op should drop by roughly the per-iteration kernel's (launch + sync) × iteration-count; after the fix, NPU compute should be a few percent of total-op and the residual should be CPU + D2H/H2D. Flavor C: total-op should drop by roughly the sum of the deleted aclnn op launches; after the fix, NPU compute should be the Triton kernel alone (a few % of total-op) and the residual should be CPU + one H2D per helper.
3. **Invocation count**: confirm the per-iteration NPU kernel's invocation count dropped to 0 (Flavor B) — the profile is the proof, not the speedup number alone. For Flavor C, confirm the many small `aclnn*` wrapper ops (aclnnArange/Linspace/Mul/Add/Clamp/Stack/Cat/Cos/Sin/Index/Repeat) dropped out of the profile — the residual NPU ops should be just the Triton kernel plus any genuinely large-data ops you kept.
4. **Boundary check (Flavor C)**: confirm at least one bulk-parallel Triton kernel remains on the NPU path (the gather/reduction kernel). If none remain, you crossed into a pure-PyTorch rewrite — back out.
5. **CPU utilization**: ensure the CPU loop does not bottleneck concurrent work.
6. **Shape diversity**: verify across all benchmark shapes — Flavor B's CPU overhead is near-constant per iteration, while the original NPU launch×count cost scaled with iteration count, so the win should hold or grow on larger cases (until D2H transfer of the per-candidate data starts to dominate). Flavor C's CPU overhead is near-constant per helper call while the original NPU launch×count scaled with grid/entry count, so the win should hold or grow on cases with more grid entries (until H2D transfer of a large helper output starts to dominate — check the largest case).

## Related Patterns

- `auxiliary-op-fusion` — fuse/ Tritonize one-shot auxiliary ops; prefer it over Flavor A when the fused NPU path is genuinely cheaper than the CPU round-trip, and over Flavor B when the loop turns out to be collapsible (state read only after the loop). **Flavor C is the fallback when `auxiliary-op-fusion` has already been tried and the residual is still a chain of small NPU wrapper ops on small data** — at that point fusing them into fewer NPU ops no longer helps (the launch overhead is the floor), and moving the whole helper to CPU is the only way past it. Do not confuse the two: fusion keeps work on the NPU; Flavor C moves it off.
- `grid-flatten-and-ub-buffering` — kernel-level batching; try it before Flavor B when the per-iteration work is large/dense enough that a batched NPU kernel could win.
- `scalar-latency-traps` — Flavor B's per-iteration NPU kernel is almost always launch/scalar-bound; this pattern explains why its per-block time is constant. Flavor C's many small `aclnn*` wrapper ops share the same root cause (each op's latency is launch/setup-bound, not compute-bound), which is why their collective overhead dominates even though no single op is large.
- `layout-materialization-elision` — another wrapper-level delete-redundant-work pattern; composes with Flavor A's scatter cleanup and with Flavor C's spatial-permute-to-index-array sub-case (Step 3).
- `algebraic-optimization` — mathematical rewrites that can further shrink the per-iteration numpy expression in Flavor B and the helper body in Flavor C.
