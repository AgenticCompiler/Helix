# Auxiliary Op Fusion

## Summary

Fuse or Tritonize simple Torch/CANN auxiliary operators when they only produce intermediate values for a downstream Triton path. The optimized path should express the target auxiliary logic in Triton; simple Torch wrapper glue is allowed, but Torch/CANN compute operators should not replace the auxiliary logic being optimized. Each external auxiliary op is a separate GM↔UB round-trip plus an AIV kernel launch; removing those external ops reduces launch count, intermediate tensor traffic, and `total_op_avg_time_us`.

## Wrapper Torch API Boundary

Simple Torch wrapper glue is acceptable when it is not the target auxiliary computation. This means buffer allocation (`torch.empty`, `torch.zeros`, `torch.ones`), shape/view preparation (`view`, `reshape`, `permute`, `transpose`, `contiguous` when required by the kernel contract), metadata reads (`shape`, `dtype`, `device`, `numel`), and Triton kernel launch setup. Do not use Torch compute functions, tensor arithmetic, reductions, or another complex Torch/CANN operator such as an NPU/CANN aggregation or a pre-baked `aclnn*`/`torch.ops.npu.*` op as the result of this pattern.

**Data rearrangement is not a bypass.** A common rationalization is to re-label a torch arithmetic chain that packs, unpacks, or reorders tensor data as "data rearrangement" or "non-compute glue" so it appears outside the Wrapper Torch API Boundary. Reordering that needs arithmetic (e.g. int4 → int8 pack via `torch.where(idx < 0, x | 0xF0, x & 0x0F)` style, or signed/unsigned conversion with masking) is a compute op, not a layout copy, and falls under the boundary. Only pure layout copies (`view`, `reshape`, `permute`, `transpose`, `contiguous`, `aclnnInplaceCopy`) are glue; anything that combines tensor values with arithmetic is the auxiliary logic itself and must be Tritonized.

When the auxiliary lookup is a small per-element indirect load (e.g. weight/bias indexed by a per-element channel or group id), keep the standard indirect `tl.load(ptr + indirect_offset)` form by default. Consider the Triton-side `tl.gather(small_table, index, axis=0)` follow-up only when the table is small enough to fit in UB and profile or simulator evidence shows the indirect load is a bottleneck — see "Gather Optimization for Small Lookup Tables" under "Tritonize Frequency Count Metadata".

**Variant dispatch across auxiliary flags**: when the operator has multiple auxiliary-flagged cases (e.g. `use_dropout` vs not), each case's wrapper path must be checked against the Wrapper Torch API Boundary independently. A common rationalization is to fuse only the flag-bearing case (e.g. dropout case) and leave the non-flag case as a torch aclnn chain (`aclnnMul + aclnnReduceSum + aclnnSub + aclnnMul*2`). This is a boundary violation for the non-flag case. Every flag branch that computes auxiliary logic must dispatch to a fused or Tritonized compute path; this may be one shared kernel with identity constants for disabled flags (e.g. `dropout_mask=1, inv_keep_prob=1`) or separate specialized Triton variants when that is simpler or faster.

## Use When

- Source code has a clear **auxiliary-op sequence -> Triton path** structure.
- The auxiliary ops compute intermediate values such as scales, masks, clamps, casts, offsets, row statistics, or broadcasted factors that are consumed by the Triton path.
- The auxiliary ops compute frequency/count metadata such as `bincount`, per-key counts, label counts, or segment counts that are consumed by the Triton path for scaling, filtering, normalization, or weighting; the output domain is known or bounded and the result is metadata rather than the operator's primary output.
- Perf output shows the auxiliary ops in `ops` before the main Triton path, and their combined time is meaningful in `total_op_avg_time_us`.
- The auxiliary output has one dominant downstream consumer, OR multiple consumers that share the same upstream load (multi-output fusion case).
- If the auxiliary output is part of the API result, the fused or Tritonized path can still store the same output.
- The auxiliary logic can be expressed in Triton with simple elementwise math, broadcast, cast, clamp, masking, row-wise reduction, scale computation, frequency count, or simple index transforms. Use Torch only for non-compute wrapper glue; do not delegate the target auxiliary computation to `torch.ops.npu.*`, `aclnn*`, or another framework compute op.
- Simulator data for the fused candidate does not show that the extra in-kernel work overwhelms the removed auxiliary-op cost.

## Avoid When

- Multiple downstream operators consume different subsets of the auxiliary output, so the fused kernel would have to recompute the shared load for each consumer. (Multi-output fusion from a single shared load is fine — see Signals.)
- The auxiliary operation has complex global semantics with order-dependent results or large/irregular output cardinality, such as sort, topk, unique, nonzero, complex gather/scatter, or cross-row/cross-batch dependencies.
- Do not treat every global-looking aggregation as complex: order-independent aggregations with bounded output such as per-key count, label count, histogram-like bins, or per-key sum can still be candidates when the result is only metadata for a downstream Triton path.
- The candidate auxiliary output is actually the operator's primary API result, not temporary metadata for a downstream Triton path; choose a result-producing kernel strategy instead of treating it as auxiliary glue.
- The auxiliary output is API-visible and cannot be produced exactly by the Triton path.
- PyTorch/CANN has special numerical behavior that is hard to reproduce in Triton, such as rounding modes, NaN/Inf behavior, dtype promotion, saturation, or broadcast corner cases.
- The auxiliary op is a pure layout copy (`aclnnInplaceCopy`, `Transpose`, `Contiguous`, `Copy`). Layout copies belong to `layout-materialization-elision`, not arithmetic fusion.
- The fused logic would be delegated to a `torch.ops.npu.*`, `aclnn*`, or Torch framework compute op instead of a Triton kernel. Simple wrapper glue is fine; swapping one framework compute op for another is not auxiliary-op fusion.
- Fusion adds more full input passes than the original auxiliary-plus-downstream sequence: i.e. the fused kernel rescans the entire input more times than the wrapper's auxiliary ops plus the downstream kernel already do. This usually means the fusion form was chosen wrong, not that the operator is unfit for fusion.
- Perf shows the auxiliary ops are tiny and the main Triton kernel is already the dominant bottleneck.
- Simulator suggests the fused kernel introduces too much register pressure, scalar/control overhead, MTE pressure, or occupancy loss.

## Regression Is Not a Permanent Rejection

A common rationalization trap is to treat one failed fusion attempt as proof that fusion is wrong for the operator. Treat the regression as evidence about the attempted form, not as a permanent rejection of the pattern.

**Precondition for this framing**: the dependency graph still admits a fusion form — i.e. there exists an auxiliary-op -> downstream Triton path where the auxiliary output is consumed by one dominant downstream kernel traversing the same base input. If the dependency graph no longer admits any fusion form (e.g. the auxiliary output now has multiple consumers with different traversal patterns), do not apply this framing; follow the "re-rank other candidate patterns" guidance in step 13 of the Implementation Sketch.

When the precondition holds and a fusion attempt regresses:

1. **Use concrete evidence to separate form failure from pattern failure.** Document whether the regression is likely caused by double reads, register/UB pressure, occupancy loss, address-generation overhead, or a correctness/compile blocker. Do not keep the unfused wrapper chain solely because of intuition about those costs.
2. **Choose at most one evidence-backed follow-up form for the current round.** If the first form regresses but shape, dependency, or simulator evidence still favors fusion, try the single most plausible alternative: single-pass compact fusion for proven small-K rows, a two-pass single-kernel form for wide rows, or shape-dispatched composition when the shape set clearly splits. Defer other variants to later rounds instead of spending the current round on a sequence of speculative benchmarks.
3. **Keep a Tritonized auxiliary path only with concrete evidence.** A preprocessing kernel plus downstream kernel is valid when shape analysis, profile data, simulator data, or measured results show single-kernel fusion is blocked by compile/correctness issues or loses because of register/UB pressure, occupancy loss, repeated reads, or another measured bottleneck. Record the specific reason.

**Failure signal that this framing itself is wrong**: if a coherent implementation plus one evidence-backed follow-up both regress on net `total-op`, and the bottleneck is no longer the removed auxiliary chain, re-rank other candidate patterns instead of continuing to force fusion in the same round.

## Signals

### Code

- A Python wrapper computes temporary tensors with `torch` or CANN-backed ops and immediately passes them into a Triton kernel.
- Common source patterns include:
  - `x.float().abs().amax(...).div(...).clamp(...)` feeding a quantization kernel.
  - `counts = torch.bincount(indices, minlength=...)` or an equivalent per-key count used only by a downstream Triton kernel, often as `counts[keys]` for scaling or filtering.
  - `mask = ...` or `offsets = ...` materialized outside a kernel and consumed once.
  - `scale`, `bias`, `smooth`, or normalization factors computed by auxiliary ops before a row-wise kernel.
- The auxiliary tensor is not used outside the local operator implementation, or it can be written by the Triton path as part of the public output contract.
- The downstream Triton kernel already loads the same base tensor or nearby metadata, so the auxiliary logic can be colocated with existing tile loops.
- The dependency is local enough for Triton tiling: elementwise, per-row, per-block, reducible within the same logical tile, or expressible as `tl.atomic_add` into a bounded metadata buffer.

### Profile

- Perf `ops` contains several small or medium auxiliary ops around the target Triton path, such as `Abs`, `ReduceMax`, `Amax`, `Div`, `RealDiv`, `Clamp`, `Cast`, `Where`, `Bincount`, or count-like framework kernels. Pure layout copies (`Copy`, `InplaceCopy`, `Broadcast` as a view) are not fusion candidates here — see `layout-materialization-elision`.
- A framework frequency-count op is a top `total_op_avg_time_us` contributor, while its output is only read by a later Triton path; as a soft signal, the auxiliary op or sequence is often worth trying when it is roughly >= 30% of `total_op_avg_time_us` or comparable to the downstream Triton kernel time.
- `total_op_avg_time_us` is materially larger than the main `kernel_avg_time_us` because auxiliary ops are counted. When `total_op_avg_time_us / kernel_avg_time_us > 2`, framework-op replacement may pay off even if the downstream kernel is already well optimized.
- Removing auxiliary ops would improve the current triton-agent metric even if pure host wall-clock gaps are not directly scored.
- Shape-level perf suggests auxiliary-op overhead dominates small or medium shapes, while large shapes may be limited by memory traffic.
- Repeated runs should be checked for cold-start or profiling outliers before attributing all speedup to fusion.
- Simulator `report.txt` shows high VGATHER UB conflicts or high SCALAR around indirect `tl.load(ptr + offset)` where `offset` is per-element computed (not a contiguous slice). This signals a candidate for `tl.gather` from a small UB-resident lookup table — see "Gather Optimization for Small Lookup Tables".

### Simulator

- Use simulator after building a fused candidate, not as the only discovery mechanism.
- A good fused candidate should not replace removed auxiliary ops with a much worse kernel profile.
- Watch for extra full input passes, increased MTE pressure, high scalar/control ratio, poor vector utilization, or larger register/UB pressure.
- If simulator shows thin row-wise programs after fusion, combine with `program-multiple-rows`.
- If simulator shows fusion is expensive only for wide shapes, consider shape dispatch between fused and unfused paths.

## Ascend Mechanics

Fusion pays off specifically because each auxiliary op becomes its own GM↔UB round-trip and AIV kernel launch. Three Ascend idioms recur in production fused kernels and should be applied when writing the fused candidate:

- **Query AIV core count explicitly and pin the grid to it.** Do not assume GPU-style SM oversubscription.
  ```python
  device_properties = triton.runtime.driver.active.utils.get_device_properties(q.device)
  num_vectorcore = device_properties.get("num_vectorcore", -1)
  grid = (num_vectorcore,)
  ```
- **Manual core-id chunking.** With the grid pinned to `num_vectorcore`, each program owns a disjoint row range rather than relying on tile-pid dispatch.
  ```python
  core_id = tl.program_id(0)
  core_num = tl.num_programs(0)
  batch_per_core = tl.cdiv(TOTAL_BATCH, core_num)
  start_batch = core_id * batch_per_core
  end_batch = tl.minimum(start_batch + batch_per_core, TOTAL_BATCH)
  ```
- **UB-aware block sizing.** Size the in-kernel tile against the per-core UB, not against an autotuner search alone.
  ```python
  MAX_FUSED_SIZE = 65536 // x.element_size()
  BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
  ```

## Optimization Notes

### Terminology

In this pattern, **auxiliary ops** are Torch/CANN-backed operations used only to prepare intermediate values for the target operator path, such as scales, masks, offsets, row statistics, clamps, casts, or broadcast factors. Older notes may call them "helper ops"; treat the two terms as equivalent, and prefer "auxiliary ops" in new guidance.

The **downstream Triton path** is the Triton kernel or Triton-kernel sequence that consumes the auxiliary output and produces the operator's main result. If an auxiliary result is part of the public API output, fusion is still valid only when the Triton path stores the same value with matching semantics.

### Choose the Implementation Form

Before picking a form, classify the code by whether the auxiliary logic and the downstream Triton path operate on the same base input and the same row/tile ownership. The examples below cover common fusion shapes; triage should match the target code against each example's applicability conditions first, then adopt the matching form. These examples are not exhaustive: if none fits but the dependency graph still has a clear auxiliary-op -> downstream Triton path, derive another fusion form only if it preserves the following invariants: remove the target Torch/CANN compute ops, avoid unnecessary GM round-trips, and prove the result with correctness plus `total-op` perf.

- **Single-kernel fusion:** inline the auxiliary logic into the consuming Triton kernel so intermediates stay in registers or UB and never round-trip through GM. Preferred when the auxiliary output is a per-row/per-tile scalar or small vector (e.g. scale, amax, mean) consumed by one downstream kernel that traverses the same base input in the same tile — see "Fuse Auxiliary Statistics Into a Downstream Kernel" applicability conditions.
- **Tritonized auxiliary path plus downstream kernel:** replace Torch/CANN auxiliary ops with a Triton preprocessing kernel while keeping the consuming downstream Triton kernel separate. Use this for count/scatter-style metadata as in "Tritonize Frequency Count Metadata", or as part of shape-dispatched composition when measured shape evidence shows single-kernel fusion loses because of register/UB pressure, repeated reads, or occupancy loss while a Tritonized auxiliary path wins `total-op`.
- **Shape-dispatched composition:** keep more than one Triton-based path when fusion wins only for some shape ranges. For example, perf comparison across shape buckets shows single-kernel fusion wins for small/medium shapes but regresses on wide shapes (or vice versa). small shapes may prefer single-kernel fusion while wide shapes prefer Tritonized preprocessing plus the optimized downstream kernel.

## Implementation Sketch

1. Identify the auxiliary sequence and the main Triton path from source code.
2. Confirm from perf that the auxiliary ops appear in `ops` and contribute meaningful `total_op_avg_time_us`.
3. Trace data dependencies:
   - Is the auxiliary output temporary?
   - Does it have one main consumer, or multiple consumers sharing the same upstream load?
   - Must it still be returned to the caller?
4. Verify the auxiliary logic is expressible in Triton, including NaN/Inf behavior, dtype promotion, saturation, and rounding modes. Simple Torch wrapper glue is allowed, but the target auxiliary computation should not be delegated to a framework compute op.
5. If the downstream Triton kernel was already optimized in the current operator file or previous local round, use that implementation as the base for the fusion attempt.
6. Choose the implementation form from the applicability conditions above:
   - If the auxiliary logic and the downstream computation belong to the same row/tile path over the same base input and have one dominant consumer, start with single-kernel fusion (the win comes from load sharing + launch removal).
   - If the auxiliary output domain is misaligned with downstream traversal, requires bounded metadata aggregation, or measured shape evidence shows single-kernel fusion loses (e.g. double-read, register/UB pressure, occupancy loss) while a Tritonized auxiliary path wins `total-op`, use a Tritonized auxiliary path.
   - If neither applies but the dependency graph still has a clear auxiliary-op -> downstream Triton path, derive a new fusion form only if it preserves the invariants in the form-selection note above; otherwise reject this pattern for the round and re-rank other patterns instead of forcing a weak fusion.
7. Set the grid by `num_vectorcore` and use manual core-id chunking, not GPU-style SM oversubscription (see "Ascend Mechanics").
8. If the auxiliary output is public, store it from the Triton path.
9. Keep the original unfused path available when needed for complex shapes, group paths, or fallback dispatch.
10. Run correctness tests for all dtype, shape, broadcast, and boundary cases.
11. Compare both `kernel` and `total-op` metrics against baseline.
12. Use simulator to inspect whether the fused or Tritonized path introduced new device-side bottlenecks.
13. If this pattern does not improve warm `total-op` after coherent attempts, or simulator data shows a new dominant bottleneck, re-rank other candidate patterns instead of continuing to force fusion.

## Example: Fuse Auxiliary Statistics Into a Downstream Kernel

A common baseline materializes row-wise preprocessing outside the downstream kernel:

```python
intermediate = some_tensor.float().abs().amax(dim=1)
scale = (intermediate / divisor).clamp(min=floor)
result = _launch_downstream_kernel(some_tensor, scale, extra_factors)
```

Perf shows several arithmetic auxiliary ops (`Abs`, `ReduceMax`/`Amax`, `Div`/`RealDiv`, `Clamp`, `Cast`) preceding the downstream kernel, all counted toward `total_op_avg_time_us`.

Applicability conditions for this form:

- The auxiliary output is a per-row/per-tile scalar or small vector (e.g. `scale`, `amax`, `mean`), with a single dominant downstream consumer.
- The auxiliary logic and the downstream computation belong to the same row/tile path over the same base input.
- For row-wise statistics with one dominant downstream consumer, prefer this form even when the kernel needs to reload the row in a second pass.
- Prefer a single Triton kernel when the statistic and downstream work can be computed together.
- If the statistic needs a second pass over the same row inside the same kernel, keep it in the same kernel unless concrete evidence shows the extra read or added in-kernel work is worse than a separate preprocessing kernel.

Failure signal: if an attempt for this code class leaves leftover Torch reduction/arithmetic in the wrapper, the implementation form is wrong. Prefer a true single-kernel fusion for row-wise statistics when the same row/tile traversal can compute the statistic and downstream result together. A separate Triton preprocessing kernel is valid only when concrete shape, profile, simulator, compile, correctness, or measurement evidence shows the single-kernel form is a poor fit. Performance concerns such as double-read cost, register-pressure estimates, or bandwidth intuition should be documented as evidence, not used as an unsupported reason to keep the wrapper chain.

The fused candidate keeps the row path in one Triton kernel. When a row is wider than one tile, each pass loops over column blocks; the statistic is accumulated into a scalar register in pass 1, and the downstream work is applied in pass 2 over the same blocks. This is shape pseudocode; adapt block sizes, masks, dtype policy, rounding, and downstream math to the actual operator:

```python
@triton.jit
def fused_row_kernel(x_ptr, out_ptr, scale_ptr, n_rows, n_cols,
                     BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    row_base = row * n_cols
    row_valid = row < n_rows

    # Pass 1: accumulate the per-row statistic (e.g. max(abs(x))) into a scalar.
    row_stat = 0.0
    for col_start in range(0, n_cols, BLOCK_N):
        offs = row_base + col_start + tl.arange(0, BLOCK_N)
        mask = row_valid & ((col_start + tl.arange(0, BLOCK_N)) < n_cols)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        row_stat = tl.maximum(row_stat, tl.max(tl.abs(x)))

    scale = tl.maximum(row_stat / 127.0, 1e-10)
    tl.store(scale_ptr + row, scale, mask=row_valid)

    # Pass 2: re-walk the same row and apply the downstream computation.
    for col_start in range(0, n_cols, BLOCK_N):
        offs = row_base + col_start + tl.arange(0, BLOCK_N)
        mask = row_valid & ((col_start + tl.arange(0, BLOCK_N)) < n_cols)
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = downstream_compute(x, scale)
        tl.store(out_ptr + offs, y, mask=mask)
```

The two passes re-read the row from GM, but they stay inside one kernel launch and never materialize the statistic or intermediate scale through GM. This is the preferred single-kernel form when the row is wider than one tile; a separate preprocessing kernel is a fallback only when concrete evidence shows the in-kernel re-read or added in-kernel work is worse.

## Example: Single-tile Compact Fusion (K fits in one tile)

A common baseline materializes row-wise preprocessing outside the downstream kernel, and the two-pass single-kernel form (see "Fuse Auxiliary Statistics Into a Downstream Kernel") is used when the row is wider than one tile. But when the row width fits in one tile, the two-pass re-read is unnecessary — the statistic can be accumulated and the downstream computation applied in a single pass.

Shape analysis, profile data, simulator data, or previous measurements may identify the two-pass form's second `tl.load` as a likely cost on small-K cases, especially when the wrapper chain is already cheap or the row fits entirely in one tile. In that case, compact fusion can be selected directly for the small-K path instead of first implementing a two-pass candidate only to prove the second load is expensive.

Applicability conditions for this form — **all must hold**:

- The row width `K` fits in one tile (`K <= BLOCK_K`), so the entire row can be loaded in a single `tl.load`. **This must be verified by host-side dispatch (`if K <= BLOCK_K`), not by agent judgment — loading a partial row in the compact form causes correctness errors.**
- The downstream computation (e.g. `result = p_a * (grad_scaled - row_sum) * mask`) is elementwise on the same tile and can be computed in the same pass — no cross-row dependency, no second reduction over a different axis.
- **Objective evidence**: shape facts, dependency analysis, simulator data, or previous measurements show that avoiding the second row load is likely to matter. A prior two-pass benchmark is useful evidence, but it is not required when the small-K path is already proven by static shape or dispatch conditions.

**Distinction from "Fuse Auxiliary Statistics Into a Downstream Kernel"**: both examples cover row-wise reduction fusion (per-row statistic + downstream apply). The difference is row width:
- "Fuse Auxiliary Statistics" — row wider than one tile, two-pass (pass 1 accumulate, pass 2 re-read + apply). Default form.
- This example — row fits in one tile (`K <= BLOCK_K`), single-pass (load once, accumulate + apply in-register, store once). Optional form for proven small-K paths when shape, profile, simulator, or measurement evidence indicates the second row load is worth avoiding.

If unsure whether the row fits in one tile, keep the two-pass form — it is correct for all row widths.

```python
@triton.jit
def _fused_row_kernel_compact(x_ptr, out_ptr, n_rows, K,
                              BLOCK_M: tl.constexpr,
                              BLOCK_K: tl.constexpr):
    # K <= BLOCK_K verified by host-side dispatch before launch
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < n_rows

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    offs = row_offs[:, None] * K + k_offs[None, :]
    valid = row_mask[:, None] & k_mask[None, :]

    # single load — entire row in one tile
    x = tl.load(x_ptr + offs, mask=valid, other=0.0).to(tl.float32)

    # accumulate statistic in-register, same pass
    row_stat = tl.sum(x * x, axis=1)   # per-row scalar in register

    # apply downstream computation in-register, same pass
    y = downstream_compute(x, row_stat)   # elementwise on the same tile

    tl.store(out_ptr + offs, y, mask=valid)
```

**Dispatch logic (shape-dispatched composition, host-side)**:
```python
if K <= BLOCK_K:
    _fused_row_kernel_compact[grid](x, out, n_rows, K,
                                    BLOCK_M=block_m, BLOCK_K=block_k)
else:
    _fused_row_kernel_two_pass[grid](x, out, n_rows, K,
                                     BLOCK_M=block_m, BLOCK_N=block_n)
```

**Failure signals — revert to the two-pass single-kernel form ("Fuse Auxiliary Statistics Into a Downstream Kernel") if any of these occur**:

- `K > BLOCK_K` (the row does not fit in one tile — the compact form cannot load the whole row at once, correctness error).
- The compact form causes UB overflow because the downstream computation needs many registers (e.g. multiple intermediate tensors alive simultaneously).
- Correctness mismatch on boundary shapes (e.g. `K == BLOCK_K` exact vs `K < BLOCK_K` masked, dtype promotion differences between masked load and unmasked load).
- Benchmark shows the compact form loses to the two-pass form on some `K <= BLOCK_K` shapes (rare, but possible when the downstream computation is heavy and the second `tl.load` is cheap due to cache residency).

**Representative shape (softmax-backward row, small K)**: the row is a backward row of width `K`. When `K <= BLOCK_K` (small-K cases), the compact form loads grad + probability/mask inputs once, computes the scaled gradient, row accumulation, difference, and final masked result, then stores in one pass. This removes both the wrapper arithmetic chain and the second row load that a two-pass form would need. For `K > BLOCK_K`, keep the two-pass fallback.

**Relationship to "Fuse Auxiliary Statistics Into a Downstream Kernel"**: both examples cover row-wise reduction fusion. This example does NOT modify the two-pass form — the two-pass form remains the default and is correct for all row widths. This example only adds an optional single-pass form for the narrow case where the row fits in one tile and shape, profile, simulator, or measurement evidence shows the second `tl.load` is likely to matter. If in doubt, keep the two-pass form.

**Relationship to `tile-selection-heuristic` and `exact-tile-no-boundary-fast-path`**: the `K <= BLOCK_K` dispatch guard composes with `exact-tile-no-boundary-fast-path` (when `K == BLOCK_K` exactly, the compact kernel can drop the `mask = k_offs < K` and become a no-boundary fast path); the host-side BLOCK_K sweep (`tile-selection-heuristic`) helps pick the BLOCK_K that maximizes `K <= BLOCK_K` coverage across the shape set.

## Example: Tritonize Data Pack at Store Time

A common baseline materializes a pack/reorder step outside the downstream kernel:

```python
quantized = _launch_downstream_kernel(...)  # produces int4/int8 values
packed = torch.where(idx < 0, quantized | 0xF0, quantized & 0x0F)
packed = packed.view(...).reshape(...)
```

Perf shows a 4-6 `aclnn*` chain (`aclnnLtScalar`, `aclnnAdds`, `aclnnSWhere`, `aclnnOr`, `aclnnAnd`, `aclnnReshape`) following the downstream kernel, all counted toward `total_op_avg_time_us`.

Applicability conditions for this form — **all must hold**:

- The auxiliary logic is **elementwise per-element** (mask + shift + OR / div + clamp + round + cast), NOT a reduction across the tile. **Exclusion**: if the auxiliary output is a per-row/per-tile scalar (e.g. `amax`, `mean`, `sum`), this case does NOT apply — use "Fuse Auxiliary Statistics Into a Downstream Kernel" (or "Single-tile Compact Fusion" when the row fits in one tile) instead. Mixing the two forms (single-pass pack for a reduction output) causes correctness errors.
- The pack/reorder happens at `tl.store` time of the downstream kernel — single-pass, no second load needed.
- The auxiliary output is consumed by exactly one downstream store (no multi-consumer scatter).

Failure signal: if the auxiliary logic requires reading neighboring elements across tile boundaries (e.g. packing two int4 from different rows into one int8), the single-pass store-time form cannot apply — restructure the tile layout first or fall back to a separate pack kernel.

```python
@triton.jit
def _pack_int4_to_int8_at_store_kernel(..., BLOCK_N: tl.constexpr):
    # downstream computation produces quantized int4 values in-register
    quantized = downstream_compute(...)  # int4 values, same tile

    # pack two adjacent int4 into one int8 at store time — single pass
    hi = quantized[::2] & 0x0F
    lo = quantized[1::2] & 0x0F
    packed = (hi << 4) | lo  # arithmetic, not layout copy

    tl.store(out_ptr + pack_offs, packed, mask=...)
```

The pack step is auxiliary logic (it combines tensor values with arithmetic masks), not layout glue. This is the store-time pack form: single-pass, elementwise-into-store, distinct from the two-pass reduction form in "Fuse Auxiliary Statistics Into a Downstream Kernel".

**Representative shape (int4 -> int8 pack at store time)**: int4 quantized output packed into int8 via `torch.where(idx < 0, x | 0xF0, x & 0x0F)` — 6+ `aclnn*` chain. Tritonize as a `_pack_int4_to_int8_kernel` (or fold into the quantization kernel) that packs two adjacent int4 into one int8 byte at `tl.store` time.

**Representative shape (quantize div+clamp+round+cast at store time)**: quantize `div + clamp + round + cast` into int8 via torch chain (`aclnnDiv` + `aclnnClamp` + `aclnnRound` + `aclnnInplaceCopy`). Same sub-form — Tritonize as a `_quantize_output_kernel` that does div+clamp+round+cast at `tl.store` time. The auxiliary logic is elementwise (no reduction), so single-pass store-time form applies.

## Example: Tritonize Frequency Count Metadata

This form is for scatter-like or bounded-metadata auxiliary paths, not for row-wise statistics that can still live in the consuming kernel.

A common wrapper-side pattern computes per-key counts only so a downstream Triton kernel can normalize or weight each row:

```python
counts = torch.bincount(keys, minlength=num_keys)
scaled = grad / counts[keys]
result = _launch_downstream_kernel(grad, keys, scaled, ...)
```

Perf may show a count-like framework op, such as `Bincount` or `Histogram`, dominating `total_op_avg_time_us`, especially when the key tensor is large. If the counts are not the public result and are only consumed by the Triton path, replace the count op with a small Triton preprocessing kernel and pass the counts buffer to the downstream kernel.

Applicability conditions for this form:

- The auxiliary output domain does not align with the input tile traversal: the key→bin map is a scatter, so the downstream kernel cannot compute the count while walking the input in a single pass.
- The auxiliary logic cannot share a `tl.load` with the downstream kernel, so inlining would not save more than a separate preprocessing kernel plus a downstream kernel.
- The output domain is known/bounded (e.g. `num_keys`, label space), so the auxiliary result can be represented as a compact metadata buffer consumed by the downstream Triton path.

Failure signal: if this form does not improve warm `total-op` on net (the new Triton count kernel time plus the downstream kernel time is not smaller than the removed framework count op), the count-to-Triton rewrite itself has no value for this operator. First retry a different fusion form only if the dependency graph still admits one that preserves the invariants above; otherwise re-rank other candidate patterns instead of forcing the preprocessing form.

```python
@triton.jit
def _count_kernel(keys_ptr, counts_ptr, n_keys, ignore_key: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_keys
    key = tl.load(keys_ptr + offs, mask=mask, other=-1)

    if ignore_key >= 0:
        keep = key != ignore_key
        mask = mask & keep
        key = tl.where(mask, key, 0)

    one = tl.full((BLOCK,), 1.0, dtype=tl.float32)
    tl.atomic_add(counts_ptr + key, one, mask=mask)
```

The wrapper may still use simple Torch glue for allocation:

```python
counts = torch.zeros((num_keys,), dtype=torch.float32, device=keys.device)
_count_kernel[(grid,)](keys, counts, keys.numel(), ignore_key, BLOCK=BLOCK_SIZE)
result = _launch_downstream_kernel(values, keys, counts, ...)
```

The downstream kernel then loads `counts[key]` directly:

```python
if SCALE_BY_COUNT:
    count = tl.load(counts_ptr + key, mask=valid, other=1.0)
    values = values / count
```

Use this form when the output domain is large enough that output-owner scanning would be too expensive. After Tritonizing the count path, if profiling or simulator evidence shows the new count kernel is atomic-bound for a small output domain, compare with `atomic-contention-owner-computes-store`. The two patterns can compose: this pattern removes the framework count op, and the owner-computes pattern may remove atomic contention if the Triton count kernel exposes it.

Validation should prioritize `total-op`, not just `kernel`, because the new count kernel may be reported as additional Triton kernel time even while it removes a much larger framework op. Verify ignored keys, invalid keys, empty inputs, count dtype conversion, count overflow assumptions, and count values consumed by the downstream path.

### Gather Optimization for Small Lookup Tables (optional follow-up, not a replacement)

**This subsection is an optional follow-up optimization that applies only when the lookup table is small. It does NOT replace the standard `tl.load(metadata_ptr + key)` form shown above — that form remains the default and is correct for large lookup tables. Apply this follow-up only when its applicability conditions are met; otherwise keep the standard indirect `tl.load`.**

Before using this follow-up, verify that the current Triton Ascend environment supports `tl.gather` with the required tensor rank, dtype, and index shape. If support is unknown, unavailable, or compilation fails, keep the standard `tl.load(metadata_ptr + key)` form and record the compile/support issue instead of forcing a gather rewrite.

When ALL of the following are true:

- The Tritonized metadata buffer is **small enough to fit in UB as one tile** (e.g. `GROUP_SIZE`, `num_heads`, label space — typically <= a few thousand entries; for `num_keys` in the tens of thousands or larger, do NOT apply this follow-up, keep `tl.load(metadata_ptr + key)`).
- The downstream kernel indexes into the metadata buffer per-element via an indirect `tl.load(metadata_ptr + key)` where `key` is per-element computed (not a contiguous slice).
- Profiling or simulator evidence shows the indirect `tl.load` is a bottleneck (e.g. high VGATHER UB conflicts, high SCALAR around address generation).

...then it is **optional** to convert the indirect `tl.load` into a `tl.gather` from a UB-resident copy of the small table.

**Before** (standard form — keep when conditions above are NOT met):
```python
key = tl.load(key_ptr + offs, mask=mask, other=0)
val = tl.load(metadata_ptr + key, mask=mask, other=0.0)   # GM indirect load
result = x * val
```

**After** (optional follow-up — only when the table is small):
```python
group_table = tl.load(metadata_ptr + tl.arange(0, TABLE_SIZE), mask=...)
key = tl.load(key_ptr + offs, mask=mask, other=0)
val = tl.gather(group_table, key, axis=0)   # UB-side gather
result = x * val
```

**Failure signals — revert to the standard `tl.load(metadata_ptr + key)` form if any of these occur**:

- The current Triton Ascend environment does not support `tl.gather` for this dtype/rank/index shape, or the candidate fails to compile.
- `TABLE_SIZE` is too large to fit in UB (UB overflow, compile error, or register spill in simulator).
- The index tensor `key` is contiguous (no indirection) — `tl.gather` adds overhead without benefit.
- Simulator shows `tl.gather` introduces VGATHER contention worse than the original indirect `tl.load`.
- Correctness mismatch on boundary shapes.

**Representative shape (group-level weight/bias lookup)**: weight/bias are group-level small tensors (`GROUP_SIZE` entries each). The gather follow-up applies because `GROUP_SIZE` is small enough for UB. For operators with large lookup tables (`num_keys` in the tens of thousands), keep the standard form.

**Relationship to the parent example**: This subsection does NOT modify the standard `tl.load(metadata_ptr + key)` lookup form prescribed by "Tritonize Frequency Count Metadata" above. The standard form remains the default. If in doubt, keep the standard form.

## Advanced Fusion Forms

### Multi-Output Fusion From One Shared Load

When multiple downstream tensors all derive from the same upstream load, write one Triton kernel that loads the inputs once and stores every downstream tensor from that single load. This is the multi-output case allowed by the "Use When" clause above and is the opposite of the "different subsets of the auxiliary output" avoid case.

A representative `@triton.jit` shape, adapted from production Ascend Triton kernels:

```python
@triton.jit
def multi_output_fused_kernel(
    a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    g_ptr, beta_output_ptr,           # two downstream outputs
    NUM_HEADS, NUM_BATCHES, beta, threshold,
    BLK_HEADS: tl.constexpr, BLK_BATCHES: tl.constexpr, ROW_ITER,
):
    i_b, i_s = tl.program_id(0), tl.program_id(1)
    COL_ITER = tl.cdiv(NUM_HEADS, BLK_HEADS)

    for row_idx in range(0, ROW_ITER):
        batch_off = i_b * ROW_ITER * BLK_BATCHES + row_idx * BLK_BATCHES + tl.arange(0, BLK_BATCHES)
        for col_idx in range(0, COL_ITER):
            head_off = col_idx * BLK_HEADS + tl.arange(0, BLK_HEADS)
            off = batch_off[:, None] * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off[None, :]
            mask = (head_off < NUM_HEADS)[None, :] & (batch_off[:, None] < NUM_BATCHES)

            # one shared load bundle
            blk_A_log = tl.load(A_log_ptr + head_off, mask=head_off < NUM_HEADS)
            blk_a = tl.load(a_ptr + off, mask=mask)
            blk_b = tl.load(b_ptr + off, mask=mask)
            blk_bias = tl.load(dt_bias_ptr + head_off, mask=head_off < NUM_HEADS)

            # both outputs derived from the same load bundle
            x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
            softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
            blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
            blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))

            tl.store(g_ptr + off, blk_g.to(g_ptr.dtype.element_ty), mask=mask)
            tl.store(beta_output_ptr + off, blk_beta_output.to(beta_output_ptr.dtype.element_ty), mask=mask)
```

The key invariant is that `blk_a`, `blk_b`, `blk_A_log`, and `blk_bias` are loaded exactly once per tile, and both `g` and `beta_output` are computed and stored from that single load bundle. Splitting this into two kernels would force a second full read of the same inputs.

### Multi-Stage Fusion Within a Single Kernel

When the auxiliary sequence and the downstream computation together span several distinct stages — for example `split_qkv + rmsnorm + rope` fused into one kernel — write the stages as sequential blocks inside the same `@triton.jit` function rather than as separate kernel launches. The point is not just "fewer kernels" but that intermediate stages stay UB-resident and never round-trip through GM.

A representative structure, adapted from production Ascend Triton kernels:

```python
@triton.jit
def multi_stage_fused_kernel(
    input_gm_ptr, q_gm_ptr, k_gm_ptr, v_gm_ptr,
    q_weight_ptr, k_weight_ptr, q_bias_ptr, k_bias_ptr,
    positions_gm_ptr, cos_sin_cache_gm_ptr,
    q_hidden_size: tl.constexpr, kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr, HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr, HALF_ROPE_DIM: tl.constexpr,
    BIAS: tl.constexpr, IS_PARTIAL_ROPE: tl.constexpr,
    num_vectorcore: tl.constexpr, batch_size_per_iter_per_vec: tl.constexpr,
):
    row_pid = tl.program_id(0)

    # Stage 1: load input once
    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    # ... load input row ...

    # Stage 2: row-wise RMSNorm statistics, kept in registers
    normalized_values = values_tmp1.to(tl.float32)
    normalized_values = normalized_values * normalized_values
    normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
    normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(..., 1)
    normalized_values = values_tmp1 * normalized_values

    # Stage 3: weight/bias application per Q and K, still in registers
    normalized_q = extract_slice(...) * q_weight_values + (q_bias_values if BIAS else 0)

    # Stage 4: RoPE applied to the in-register normalized+weighted tensors
    # ... cos/sin gathered once per position, then rotate ...

    # Stage 5: store Q, K, V outputs in their final layouts
    tl.store(q_gm_ptr + q_output_idx, values_tmp.reshape(...), mask=mask)
    # ... store K, V ...
```

The crucial property is that `values_tmp1` loaded in stage 1 feeds stages 2 through 4 without ever being written to GM and re-read. Splitting `rmsnorm` or `rope` into auxiliary kernels would each force a full GM round-trip of the row.

This shape uses `extract_slice` / `insert_slice` from `triton.language.extra.cann.extension` because triton-ascend's register layout is NZ/ZZ-fractal and naive `tl.reshape` does not always compose — see `slice_coalesce` and `slice_intermediate` for the dedicated treatment.

## Related Patterns

- `algebraic-optimization`
- `atomic-contention-owner-computes-store`
- `layout-materialization-elision`
- `program-multiple-rows`
- `scalar-vector-simulation-signal`
- `slice_coalesce`
- `slice_intermediate`
- `tiling`

## What To Verify After Applying

- Correctness against the original implementation, including dtype conversion, rounding, saturation, clamp limits, NaN/Inf behavior, broadcasting, empty or tail shapes, and API-visible outputs.
- Perf: auxiliary ops disappear or shrink in `ops`, and `total_op_avg_time_us` improves against a warm and comparable baseline.
- Kernel metric: the fused kernel does not regress enough to erase the removed auxiliary-op time.
- Simulator: no new dominant bottleneck from double-read, scalar-heavy control, register/UB pressure, or poor vector utilization.
- Shape sensitivity: small and medium shapes may benefit while very wide shapes regress; use guarded dispatch when the tradeoff is shape-dependent.
