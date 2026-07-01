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

## Default Workflow

1. Print the AscendC source for an operator:
   ```bash
   python3 ./scripts/capture_ir.py --operator-file matmul.py
   ```

2. Save to a file for inspection or round archival:
   ```bash
   python3 ./scripts/capture_ir.py --operator-file opt-round-1/opt_matmul.py > opt-round-1/ir/ascendc_source.cpp
   ```

3. If the operator has multiple kernels, list them all first:
   ```bash
   python3 ./scripts/capture_ir.py --operator-file matmul.py --kernel func
   ```

4. Analyze the AscendC source for performance signals using standard terminal tools:
   ```bash
   # Count compute vs transfer operations
   rg -c 'Mmad|Muls|Add|Sub|Max|Min|Exp|Ln|Sqrt|Rsqrt' ascendc_source.cpp
   rg -c 'CopyIn|CopyOut|DataCopy' ascendc_source.cpp

   # Find synchronization points
   rg -n 'pipe_barrier|SetFlag|WaitFlag' ascendc_source.cpp

   # Check buffer allocation sizes
   rg -n 'LocalTensor' ascendc_source.cpp
   ```

5. For round-over-round comparison, archive the AscendC source under each round's `ir/` directory and diff:
   ```bash
   diff opt-round-1/ir/ascendc_source.cpp opt-round-2/ir/ascendc_source.cpp
   ```

## Working Rules

- Prefer `python3 ./scripts/capture_ir.py --operator-file ...` to extract AscendC source rather than ad hoc Python snippets.
- Archive AscendC source under `opt-round-N/ir/` for round-level evidence, mirroring the triton-npu-analyze-ir convention.
- Keep the archived AscendC source immutable once captured unless the user explicitly asks to replace it.
- Use `rg`, `grep`, `diff`, or similar terminal tools on the captured source for analysis.
- Present analysis in terms of concrete AscendC code patterns (pipeline stages, buffer sizes, sync density, compute-to-transfer ratio), not only intuition.
- If the user also needs hotspot evidence or operator timing attribution, use the `ascend-npu-profile-operator` skill as a companion skill.
