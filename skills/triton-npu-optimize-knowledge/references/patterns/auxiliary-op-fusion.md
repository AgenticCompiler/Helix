# Auxiliary Op Fusion

## Summary

Fuse or Tritonize simple Torch/CANN auxiliary operators when they only produce intermediate values for a downstream Triton path. The optimized path should express the target auxiliary logic in Triton; simple Torch wrapper glue is allowed, but Torch/CANN compute operators should not replace the auxiliary logic being optimized. Each external auxiliary op is a separate GM↔UB round-trip plus an AIV kernel launch; removing those external ops reduces launch count, intermediate tensor traffic, and `total_op_avg_time_us`.

## Wrapper Torch API Boundary

Simple Torch wrapper glue is acceptable when it is not the target auxiliary computation. This means buffer allocation (`torch.empty`, `torch.zeros`, `torch.ones`), shape/view preparation (`view`, `reshape`, `permute`, `transpose`, `contiguous` when required by the kernel contract), metadata reads (`shape`, `dtype`, `device`, `numel`), and Triton kernel launch setup. Do not use Torch compute functions, tensor arithmetic, reductions, or another complex Torch/CANN operator such as an NPU/CANN aggregation or a pre-baked `aclnn*`/`torch.ops.npu.*` op as the result of this pattern.

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
- Fusion turns a single-pass kernel into a much more expensive multi-pass kernel for memory-bound or very wide shapes.
- Perf shows the auxiliary ops are tiny and the main Triton kernel is already the dominant bottleneck.
- Simulator suggests the fused kernel introduces too much register pressure, scalar/control overhead, MTE pressure, or occupancy loss.

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

- **Single-kernel fusion:** inline the auxiliary logic into the consuming Triton kernel so intermediates stay in registers or UB and never round-trip through GM.
- **Tritonized auxiliary path plus downstream kernel:** replace Torch/CANN auxiliary ops with a Triton preprocessing kernel while keeping the consuming downstream Triton kernel separate. Use this when single-kernel fusion would force expensive double reads or high register/UB pressure.
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
6. Choose the implementation form:
   - Inline the auxiliary logic into the consuming Triton kernel when load sharing and launch removal dominate.
   - Write a Triton auxiliary/preprocessing kernel and keep the downstream kernel separate when single-kernel fusion causes double-read or register/UB pressure.
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

The fused candidate:

- One pass: load the row, compute the intermediate statistic (e.g. `max(abs(x))`), derive `scale` from it, then apply the downstream computation and store both the API-visible `scale` and the final result.
- For wider shapes where the statistic and downstream computation cannot share a single pass, split into two passes inside the same kernel and reload the row, accepting the double read in exchange for eliminating auxiliary-op launches.
- Follow up with `program-multiple-rows` to amortize per-program fixed cost if the fused kernel becomes launch-bound rather than memory-bound.

This removes the external auxiliary ops for the simple path while keeping a fallback dispatch available for shapes where the double read dominates.

## Example: Tritonize Frequency Count Metadata

A common wrapper-side pattern computes per-key counts only so a downstream Triton kernel can normalize or weight each row:

```python
counts = torch.bincount(keys, minlength=num_keys)
scaled = grad / counts[keys]
result = _launch_downstream_kernel(grad, keys, scaled, ...)
```

Perf may show a count-like framework op, such as `Bincount` or `Histogram`, dominating `total_op_avg_time_us`, especially when the key tensor is large. If the counts are not the public result and are only consumed by the Triton path, replace the count op with a small Triton preprocessing kernel and pass the counts buffer to the downstream kernel.

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
