# Argsort Avoid AiCPU Fallback

## Summary

When Ascend `ArgSort` would fall back to AiCPU for `int32` or `int64` keys, cast the keys to `float32` first so the sort can stay on AiCore, but only when the integer domain is exactly representable in `float32`.

## Use When

- The hot path calls **`torch.argsort()`** on **`int32`** or **`int64`** tensors on Ascend NPU.
- Runtime logs report that **`ArgSort`** does not support the active integer dtype on **AiCore** and is running on **AiCpu** instead.
- The sort keys are small-range integers such as **expert ids**, **routing ids**, **bucket ids**, or similar categorical keys whose absolute values stay within the exact-integer range of **`float32`**.
- Profiling shows unusually high sort latency for tiny or moderate element counts, consistent with fallback dispatch overhead rather than the logical sort size.

## Avoid When

- Integer keys may exceed the exact integer range of **`float32`** (for example, large global ids, hash keys, or token ids above **`2^24`** in magnitude).
- The input is already **`float32`** or another supported dtype, so the cast would not change backend placement.
- The hot operator is not clearly **`ArgSort`**. Do not assume **`topk`** or other selection operators share the same kernel support path without matching warning or profiler evidence.
- Ordering semantics depend on preserving distinctions between large neighboring integers that would collapse after a **`float32`** cast.

## Signals

### Code

- **`torch.argsort(x, stable=...)`** is called on an **`int32`** or **`int64`** tensor.
- The code uses a dtype- or shape-gated branch that casts to **`float32`** only for some cases, leaving smaller cases on integer sort.
- The sort keys come from bounded integer generation or routing metadata such as **`torch.randint(0, num_experts, ...)`**.

### Profile

- `msprof` or similar profiling shows **Sort / ArgSort** latency in the **100us+** range even when the sorted length is tiny.
- A **`float32`** sort on a larger case is still much faster than the integer-dtype sort on a smaller case, which indicates fallback overhead rather than sort volume is dominating.

## Related Patterns

- `algebraic-optimization`
- `vec-cmp`

## What To Verify After Applying

- **Semantic equivalence:** every key value remains exactly representable after the **`float32`** cast, so ordering and **`stable=True`** behavior are unchanged for unequal keys.
- **Backend placement:** the runtime warning disappears or profiling confirms the sort now stays on **AiCore** instead of **AiCpu**.
- **End-to-end cost:** the added cast cost is much smaller than the removed fallback overhead on representative shapes.
- **Scope:** only the targeted `ArgSort` path changes; unrelated operators do not silently inherit the cast without evidence.

## Problem Description

On Ascend NPU, the runtime may not support **`ArgSort`** on **`int32`** or **`int64`** in **AiCore**. In that situation, the framework falls back to **AiCpu** and may emit a warning like:

```text
kernel [ArgSort] can not support dtype int32 or int64 on AiCore,
Now this kernel is running on AiCpu.
If you are more concerned about high-performance execution,
please cast dtype to float32.
```

This is not fully silent because the warning exists, but it is easy to miss in normal logs. The performance signature is often more surprising than the warning itself: very small sorts still cost hundreds of microseconds because fallback dispatch, framework scheduling, and synchronization dominate.

## Optimization Strategy

1. Confirm the hot operator is **`torch.argsort()`** on an integer tensor.
2. Confirm the keys are in the exact integer range of **`float32`**.
3. Cast the keys to **`float32`** immediately before the sort.
4. Re-profile to confirm the sort remains on **AiCore** and the cast overhead is negligible relative to the removed fallback.

## Example

Before:

```python
if route_count >= 2048:
    sorted_indices = torch.argsort(expert_flat.to(torch.float32), stable=True)
else:
    sorted_indices = torch.argsort(expert_flat, stable=True)
```

After:

```python
sorted_indices = torch.argsort(expert_flat.to(torch.float32), stable=True)
```

This rewrite is valid only when the integer key domain is exactly representable in **`float32`**. Small MoE expert ids usually satisfy that condition; large global integer ids may not.
