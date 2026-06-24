# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.

## High Priority Patterns

- None.

## Generated Pattern Summaries

### `argsort-avoid-aicpu-fallback`

- Summary: When Ascend `ArgSort` would fall back to AiCPU for `int32` or `int64` keys, cast the keys to `float32` first so the sort can stay on AiCore, but only when the integer domain is exactly representable in `float32`.
- Source: [argsort-avoid-aicpu-fallback.md](patterns/argsort-avoid-aicpu-fallback.md)
- Use When:
  - The hot path calls **`torch.argsort()`** on **`int32`** or **`int64`** tensors on Ascend NPU.
  - Runtime logs report that **`ArgSort`** does not support the active integer dtype on **AiCore** and is running on **AiCpu** instead.
  - The sort keys are small-range integers such as **expert ids**, **routing ids**, **bucket ids**, or similar categorical keys whose absolute values stay within the exact-integer range of **`float32`**.
  - Profiling shows unusually high sort latency for tiny or moderate element counts, consistent with fallback dispatch overhead rather than the logical sort size.
