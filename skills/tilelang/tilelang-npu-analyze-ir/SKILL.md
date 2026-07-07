---
name: tilelang-npu-analyze-ir
description: Use when an agent needs to inspect generated AscendC source code from a TileLang compiled operator, reason about likely performance issues from the generated code, or compare AscendC output across optimization rounds.
---

# Ascend Operator IR Analyzer (TileLang)

## Overview

Extract and inspect the generated AscendC source code from a TileLang compiled operator. TileLang compiles Python DSL kernels into AscendC source via `get_kernel_source()`, which is then compiled by Bisheng into a `.so` for execution on the NPU.

```
Python DSL → TIR (lowering) → AscendC CodeGen → Bisheng → .so
                                  ↑
                             capture_ir.py extracts this
```

## Prerequisites

- `capture_ir.py` depends on `@tilelang.jit` finishing compilation during module import.
- Trigger that compilation with a module-level call such as `compiled_kernel = kernel_func(...)`.
- Do not rely on `_ = kernel_func(...)` or any other exported name that starts with `_`. Even if that import triggers compilation, `capture_ir.py` skips module-level names that start with `_`.

## Default Workflow

1. Clear stale caches before capture:
   ```bash
   find <workspace> -name "__pycache__" -type d -exec rm -rf {} +
   ```
   If a previous JIT attempt failed, also remove stale TileLang memoize artifacts such as `.pkl_memoize_py3` before retrying the same operator and parameters.

2. Print the AscendC source for an operator:
   ```bash
   python3 ./scripts/capture_ir.py --operator-file matmul.py
   ```

3. Save to a file for inspection or round archival:
   ```bash
   python3 ./scripts/capture_ir.py --operator-file opt-round-1/opt_matmul.py > opt-round-1/ir/ascendc_source.cpp
   ```

4. If the operator has multiple kernels, list them all first:
   ```bash
   python3 ./scripts/capture_ir.py --operator-file matmul.py --kernel func
   ```

5. Analyze the AscendC source for performance signals using standard terminal tools:
   ```bash
   # Count compute vs transfer operations
   rg -c 'Mmad|Muls|Add|Sub|Max|Min|Exp|Ln|Sqrt|Rsqrt' ascendc_source.cpp
   rg -c 'CopyIn|CopyOut|DataCopy' ascendc_source.cpp

   # Find synchronization points
   rg -n 'pipe_barrier|SetFlag|WaitFlag' ascendc_source.cpp

   # Check buffer allocation sizes
   rg -n 'LocalTensor' ascendc_source.cpp
   ```

6. For round-over-round comparison, archive the AscendC source under each round's `ir/` directory and diff:
   ```bash
   diff opt-round-1/ir/ascendc_source.cpp opt-round-2/ir/ascendc_source.cpp
   ```

## Troubleshooting

| Error | Likely cause | Fix |
| --- | --- | --- |
| `No compiled kernels found` | The operator never triggers `@tilelang.jit` during module import | Add a module-level call such as `compiled_kernel = kernel_func(...)` |
| `No compiled kernels found` after adding a trigger call | The compiled kernel is only exported through a name that starts with `_` | Rename the exported compiled-kernel variable so it does not start with `_` |
| `Compilation Failed` | A stale failed JIT result is being reused from cache | Clear `__pycache__` directories and stale `.pkl_memoize_py3` memoize files, then retry |

## Working Rules

- Prefer `python3 ./scripts/capture_ir.py --operator-file ...` to extract AscendC source rather than ad hoc Python snippets.
- Clear stale `__pycache__` directories and `.pkl_memoize_py3` files before retrying a previously failed compilation with the same inputs.
- Archive AscendC source under `opt-round-N/ir/` for round-level evidence, mirroring the triton-npu-analyze-ir convention.
- Keep the archived AscendC source immutable once captured unless the user explicitly asks to replace it.
- Use `rg`, `grep`, `diff`, or similar terminal tools on the captured source for analysis.
- Present analysis in terms of concrete AscendC code patterns (pipeline stages, buffer sizes, sync density, compute-to-transfer ratio), not only intuition.
- If the user also needs hotspot evidence or operator timing attribution, use the `ascend-npu-profile-operator` skill as a companion skill.
