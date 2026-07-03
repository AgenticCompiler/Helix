# TileLang Repair Experience (Ascend)

Actionable repair heuristics discovered during real TileLang Ascend NPU operator conversions. Each entry maps a symptom to the smallest fix, with cross-references to the API reference docs (kernel-basics.md, compute-developer.md) and the convert skill (SKILL.md) for fuller context.

---

## 1. Kernel Return Value Discard

**Symptom**: All test cases fail. Output tensor is all-zeros or uninitialized. The kernel compiled without error and launched, but the forward method returns empty data.

**Diagnosis**: `@tilelang.jit(out_idx=[...])` returns new tensors for marked output indices. The `forward()` method called `kernel(input, output)` without capturing the return value, so it reads back the still-empty pre-allocated buffer.

**Fix**: Change `kernel(x, y)` to `y = kernel(x, y)`.

**Cross-ref**: kernel-basics.md § Parameter Marking: `out_idx`.

**Real case**: Both `expand_kenel_fwd` and `expand_kenel_bwd` had this exact bug. Each went through 5–6 debug iterations before the root cause was identified. `expand_kenel_bwd` had a second compounding issue (threads=2 — see pattern 2).

---

## 2. threads=2 Data Race with T.serial on vid-split dim

**Symptom**: Large deterministic numerical errors (not random). Same max_abs_diff value at same position every run. The kernel compiles and launches without error, but results are wrong by wide margins.

**Diagnosis**: With `threads=2` (vid elimination), the compiler auto-splits **only** `T.Parallel` iteration spaces across the two vector cores. `T.serial` is NOT split — both cores independently execute the exact same loop body. If that loop writes to global memory, both cores write the same addresses simultaneously → data race. The buffers themselves are NOT physically split (each core gets the full buffer); it's the iteration space that stays unsplit for `T.serial`.

**Fix**: Either rewrite the `T.serial` loop on the vid-split dimension as `T.Parallel` (letting the compiler split it), or switch to `threads=1`. Note: `T.serial` on a non-split dimension (e.g. K loop in GEMM) is safe with `threads=2`.

**Cross-ref**: kernel-basics.md § Vid Elimination.

**Real case**: `expand_kenel_bwd` used `threads=2` with `T.serial(block_n1)` iterating the first (vid-split) dimension of `src_2d`, with each iteration doing `T.reduce_sum` + `T.copy` to output. Both cores executed the identical full loop, racing on `grad[...]` writes. The output was deterministic but wrong (max_abs_diff = 10.05 at the same position every run). Switching to `threads=1` resolved it — the safety table is: `T.serial(K)` ✅, `T.serial(block_M)` on split dim ❌, `T.Parallel(block_M,...)` ✅.

---

## 3. Python float Constants Downcast to Buffer dtype

**Symptom**: Kernel compiles and runs without error, but produces silently wrong results (e.g., a clamp at `eps=1e-10` has no effect). The constant appears to be ignored.

**Diagnosis**: TIR parser folds Python float literals to float32. Inside `T.Parallel`, float32 constants are implicitly cast to the buffer's element dtype — if the buffer is float16, `1e-10` underflows to `0`, making operations like `max(x, eps)` silently return `x`. It is NOT a miscompile; the constant lowered correctly, just to a value that is zero in the target dtype.

**Fix**: Pass constants as kernel parameters or pre-compute them in the `@tilelang.jit` factory function at the intended dtype so they enter the kernel as float32 or float16 values directly, rather than going through implicit float64→float16 downcast.

**Cross-ref**: compute-developer.md § T.Parallel (Restrictions).

**Real case**: `act_quant_kernel` — `eps=1e-10` (float64) was used in `T.max(absmax, eps)` inside a float16 buffer. The constant underflowed to `0.0`, making the eps clamp ineffective. The fix was not about the loop construct but about controlling the constant's dtype before it entered the kernel.

---

## 4. fp8 Types Not Available on Ascend

**Symptom**: The original operator uses `torch.float8_e4m3fn` or `torch.float8_e5m2`. TileLang's supported dtype list does not include any float8 variant. Attempting to declare a `T.Tensor` with `dtype="float8_e4m3fn"` fails at kernel definition.

**Diagnosis**: TileLang Ascend supports `float16`, `bfloat16`, `float32`, `int8/16/32/64`, `uint8/16/32/64`. No float8. Ascend hardware may support fp8 compute, but the TileLang compilation path does not expose it.

**Fix**: Compute the quantization math in float32 inside the TileLang kernel (absmax, scale, division, clamp to fp8 range). In `forward()`, apply `.to(fp8_dtype)` on the kernel output. The fp8 cast in forward is a metadata-level storage conversion, not heavy compute — acceptable when TileLang cannot express the destination dtype.

**Real case**: `act_quant_kernel` — the kernel computes `x_q` in float32 with fp8-range clamping, then `forward()` does `.clamp(fp8_min, fp8_max).to(dtype).reshape(x.shape)`.

---

## 5. Reduction Precision Loss (bf16/float16)

**Symptom**: The kernel compiles and runs, outputs are structurally correct (no NaN, right shape), but the differential test shows errors larger than the NPU accuracy contract allows — especially larger for inputs with wider value ranges.

**Diagnosis**: `T.reduce_sum` accumulates in the buffer's element dtype. float16 accumulation loses precision vs. PyTorch's internal float32 accumulation. `T.reduce_max` and `T.reduce_min` compare elements directly — no accumulation error, so they don't need float32 for precision (though matching src/dst dtypes is still required).

**Fix**: Use `"float32"` buffers for `T.reduce_sum`. Two patterns both work:
```python
# Option A: upcast in forward() — simpler
x_compute = x.float()
result = kernel(x_compute, ...)
return result.to(x.dtype)

# Option B: float32 intermediate buffer in kernel — avoids extra GM round-trip
# Declare T.alloc_shared((M, N), "float32"), T.copy from float16 GM auto-converts
```

**Cross-ref**: compute-developer.md § Reductions (Precision note).

**Real case**: `expand_kenel_bwd` — float16 inputs produced large errors in the sum reduction over `mult=4` elements. Upcasting to float32 and using `dtype="float32"` in the kernel fixed it.

---

## 6. pass_configs Key Name Mismatch

**Symptom**: `AttributeError: type object 'PassConfigKey' has no attribute 'TL_ASCEND_AUTO_CROSS_CORE_SYNC'` or the key is silently ignored (no error but cross-core sync is not enabled, causing data races).

**Diagnosis**: The correct key name is `TL_ASCEND_AUTO_CV_SYNC`, not `TL_ASCEND_AUTO_CROSS_CORE_SYNC`.

**Fix**: Use the exact key names from the reference:
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

**Cross-ref**: kernel-basics.md § PassConfigKey Reference.

---

## 7. T.Kernel "as (cid,)" Syntax Error

**Symptom**: `TypeError: cannot unpack non-iterable Var object` at the `with T.Kernel(...)` line. The traceback points to the `as (cid,)` clause.

**Diagnosis**: The trailing comma in `as (cid,)` makes Python expect a 2-element tuple, but the `T.Kernel` context manager returns a single value when `threads=2`.

**Fix**: Use `as cid` (simple variable, works universally) or `as (cid)` (single-element tuple without trailing comma).

**Cross-ref**: kernel-basics.md § Kernel Launch Variants.

---

## 8. VEC Alignment Crash on Broadcast Loads

**Symptom**: Kernel launch fails with an error containing "alignment", "broadcast", or "VEC" in the message. Happens when loading a value that needs to be broadcast across vector lanes.

**Diagnosis**: The Ascend vector engine requires aligned memory access patterns for broadcast loads. A `T.Parallel` loop that broadcasts a scalar from one buffer lane to all elements of another buffer may trigger a VEC alignment fault.

**Fix**: Try nesting `T.serial` (outer) + `T.Parallel` (inner) for the broadcast dimension. Note: this is a heuristic observed on one kernel, not a documented compiler guarantee — if it doesn't help, the alignment issue may have a different root cause.
```python
for i in T.serial(block_N):
    for j in T.Parallel(group_size):
        x_ub[i, j] = x_ub[i, j] / scale_ub[i]
```

**Real case**: `act_quant_kernel` — the division `x_ub[i, j] / scale_ub[i]` inside a flat `T.Parallel(block_N, group_size)` crashed. Nesting it as `T.serial(block_N)` outer + `T.Parallel(group_size)` inner resolved the crash.

---

## 9. Multi-dim T.copy Index Confusion

**Symptom**: Compilation error mentioning "rank mismatch", "dimension count", or "cannot infer copy extent". Typically occurs when the source tensor rank differs from the destination buffer rank.

**Fix**: Flatten the source tensor before passing to the kernel so that source and destination ranks match. Avoid multi-dimensional `T.copy` slices that mix fixed and ranged dimensions — the lowering behavior is not well-documented and varies across compiler versions.

**Real case**: `expand_kenel_bwd` — the agent spent many iterations trying to slice a 4D `o_grad` tensor into a 3D `src_2d` buffer. The eventual fix flattened `o_grad` to 2D before passing to the kernel, eliminating the rank mismatch.
