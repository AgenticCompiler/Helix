---
name: tilelang-npu-convert-pytorch-operator
description: Convert one PyTorch operator into a TileLang NPU-backed PyTorch operator, preserve the trailing input-helper block, and validate the converted output through standalone or differential testing.
---

# Convert PyTorch Operator

Convert one PyTorch operator file into a PyTorch-facing operator backed by a real TileLang Ascend NPU kernel path.

Use this skill when the user wants a new converted operator artifact instead of an in-place optimize round.

## Inputs

- one original PyTorch operator file
- one requested output path for the converted operator
- one requested standalone or differential test mode
- optional remote execution context from the outer task

## Outputs

- one converted operator file, usually named `tilelang_<origin-name>.py`
- preserved trailing input-helper block in the converted output
- one generated standalone or differential test file for the converted output
- a short summary of what was converted, what remained unchanged, and any blockers

## Core Constraints

- Treat the original input operator file as immutable source material and, for differential validation, the correctness oracle.
- Do not modify, or overwrite the original input operator file.
- Keep the public API PyTorch-facing when needed, but keep the converted computation on a real TileLang Ascend NPU kernel path.
- Target Ascend NPU only for this conversion flow; do not add CUDA, CPU, MPS, or generic multi-backend fallback logic unless the source file already requires shared import structure around the public API.
- Do not introduce unnecessary wrappers, compatibility branches, helper layers, or standalone or differential test code inside the converted operator file.

## API Hierarchy

TileLang-Ascend APIs are organized in three layers from base to advanced.

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: Expert programming model                            │
│  · Explicit hardware memory: T.alloc_ub / T.alloc_L1 / L0*   │
│  · Manual sync: T.barrier_all / T.set_flag / T.wait_flag     │
│  · pass_configs: auto-passes OFF for full manual control     │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: Extended compute primitives                         │
│  · T.tile.add / .exp / .sqrt / .cast / .compare / .select    │
│  · T.tile.sort / .topk / .merge_sort / .transpose / .fill    │
│  · Use when Layer 1 cannot express the operation             │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: Base primitives — recommended default               │
│  · Memory: T.alloc_shared / T.alloc_fragment / T.alloc_var   │
│  · Data: T.copy                                              │
│  · Compute: T.gemm_v0 / T.reduce_* / T.Parallel + sym math   │
│  · Schedule: T.serial / T.Pipelined / T.Persistent           │
│  · Infrastructure: T.Kernel / @T.prim_func / @tilelang.jit   │
│  · Auto-managed: pass_configs (AUTO_CV/SYNC/MEMORY/CROSS)    │
└──────────────────────────────────────────────────────────────┘
```

> **Note on layer naming**: This three-layer split is an engineering convention for convert workflows — a practical guide for what to reach for first, next, and last. In the TileLang source architecture, `T.tile.*` extensions and explicit memory / manual sync are all part of "Expert mode". The split here separates them into Layer 2 (extended compute — usable with auto-managed `pass_configs`) and Layer 3 (explicit hardware control — requires manual sync and Expert `pass_configs`) because they have different complexity costs in practice.

### Convert Guidance

1. **Start with Layer 1**: Use `T.alloc_shared` / `T.alloc_fragment` for memory, `T.Parallel` + symbolic math for element-wise, `T.gemm_v0` for matrix multiply, `T.reduce_*` for reductions. The compiler handles CV splitting, sync, and memory planning automatically via `pass_configs`.
2. **Use Layer 2 when needed**: When Layer 1 primitives cannot express an operation (e.g., sort, topk, compare, cast, gather_mask, transpose), use the corresponding `T.tile.*` extension.
3. **Do not use Layer 3 (Expert) in convert**: `T.alloc_ub` / `T.alloc_L1` / `T.alloc_L0*`, manual sync, and Expert `pass_configs` are for optimize tuning — not convert. Convert produces a correct baseline; Expert-level control belongs in later optimization rounds.

### Reference Documents

See the [TileLang API reference](../tilelang-npu-api-reference/SKILL.md) for all TileLang Ascend NPU API documentation.

## Required Workflow

1. Read the original operator file carefully before editing anything, and identify the public PyTorch entrypoint that should remain visible after conversion.
2. Write the converted operator to the requested output path. Keep the delivered result PyTorch-facing when needed, but move the converted computation onto a real TileLang Ascend NPU kernel path. You may replace some operators, leave some unchanged, fuse operations, or make targeted algorithmic changes when that helps the TileLang NPU path.
3. Preserve the trailing input-helper block from the source file in the converted output because later harness generation and validation may need it.
4. Validate the converted output with the requested standalone or differential mode by following the validation flow below.
5. Finish only after validation passes or a clear environment blocker prevents further progress.

## Validation Flow

1. If a suitable test already exists in the operator workspace, reuse it. This includes existing standalone and differential test cases when they already cover the operator workspace.
2. Do not create a new test when an existing suitable test can be reused unless the user explicitly asks to regenerate it.
3. When no suitable reusable test exists, use `ascend-npu-gen-test` to generate a test for the converted output.
4. Use the original input operator as the reference implementation when the requested mode is differential, and use the converted output as the system under test in all cases.
5. Use `ascend-npu-run-eval` to execute validation — run `run-test-convert` for both standalone and differential convert validation as prescribed by the skill. The command output must contain `PASS:` for validation to be considered successful. See "Validation Enforcement Rules" below for the complete set of mandatory validation constraints.
6. If the converted output hits TileLang compile, JIT, launch, or kernel-structure errors, use `tilelang-npu-repair-guide` for operator-side repair heuristics and then re-run validation.

## Validation Enforcement Rules

Correctness validation is **not advisory** — it is the final gate before conversion is considered complete. The following rules are mandatory and non-negotiable.

### Mandatory Validation Commands

You MUST use the validation commands prescribed by the `ascend-npu-run-eval` skill:

- **Differential mode**: `cli.py run-test-convert` with `--ref-operator-file <original>`
- **Standalone mode**: `cli.py run-test-convert`

These commands handle result archiving, NPU synchronization, and comparison — all of which you cannot replicate reliably with ad-hoc scripting.

### Forbidden Validation Practices

The following self-validation patterns are **strictly forbidden**. Violating any of these means the conversion is incomplete, regardless of what other checks you believe have passed.

| Forbidden | Example | Why |
|-----------|---------|-----|
| Ad-hoc `python3 -c "import torch; ..."` comparison scripts | `python3 -c "import torch; ref=torch.load(...); torch.allclose(...)"` | Bypasses the standard `compare_result.py` and its audited tolerance levels |
| Using `torch.allclose` / `torch.testing.assert_close` with custom tolerances | `torch.allclose(a, b, rtol=1e-5, atol=1e-8)` | Tolerances different from the NPU accuracy contract thresholds may mask real errors or produce false positives |
| Running a differential test file directly with `python3` instead of through `cli.py` | `python3 differential_test_xxx.py --operator-file xxx` | Bypasses result archiving, synchronization, and comparison logic |
| Generating and comparing custom `.pt` files on your own | `torch.save(ref, "REFERENCE_RESULT.pt")` then manual compare | Creates files that may interfere with the CLI validation loop and use inconsistent formats |
| Self-declaring "PASS" based on your own comparison logic | "PASS: Minor 1-ULP differences only (expected for cross-implementation)" | Only `run-test-convert` output determines pass/fail |

### What "PASS" Means

- The literal output `PASS: all N case(s) matched the NPU accuracy contract.` (or per-case `PASS case '...' matched ...`) — differential mode conversion is definitively validated
- The literal output `All tests passed!` or matching `All tests passed!` in the command output — standalone mode has passed
- ANY other output (including `FAIL:`, Python exceptions, or empty output) — conversion has NOT passed
- If the command does not print `PASS:` or `All tests passed!`, do NOT declare success. Instead, use the failure output to diagnose and fix the kernel, then re-run the same validation command.

## Converted Example

Use a real converted output example in the generated file, not only a prose description. For a simple elementwise add conversion, a converted output may look like this:

```python
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

M = 1024
N = 1024
block_M = 128
block_N = 128


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def tilelang_add(M: int, N: int, block_M: int, block_N: int, dtype: str = "float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M, block_N), dtype)
            b_ub = T.alloc_shared((block_M, block_N), dtype)
            c_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(B[bx * block_M, by * block_N], b_ub)

            for i, j in T.Parallel(block_M, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M, by * block_N])

    return add_kernel


func = tilelang_add(M, N, block_M, block_N)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return func(a, b)


def get_inputs():
    a = torch.randn(M, N, device="npu")
    b = torch.randn(M, N, device="npu")
    return [a, b]


def get_init_inputs():
    return []
```

In this kind of conversion:

- `tilelang_add(...)` is the factory function that builds and returns a compiled TileLang kernel
- `add_kernel` is the `@T.prim_func` kernel definition — all compute happens here
- `func = tilelang_add(M, N, block_M, block_N)` compiles the kernel at module load time via `@tilelang.jit`
- `class Model` is the converted public architecture — `forward()` just calls the compiled kernel
- The trailing `get_init_inputs()` / `get_inputs()` block is preserved in the converted output instead of being dropped
- The original source operator remains the correctness oracle for differential validation
- This example uses **Layer 1 (Developer mode)**: `T.alloc_shared` for memory, `T.Parallel` + symbolic math for compute, auto-managed `pass_configs` with all four auto-passes enabled, and `threads=2` vid elimination

## Forward Method Constraints

The converted operator **must** be a pure TileLang Ascend implementation. The `forward()` method may only call a pre-compiled TileLang kernel — all computation must happen inside `@T.prim_func` kernels.

### Forbidden in forward()

| Category | Examples | Reason |
|----------|----------|--------|
| `torch` compute functions | `torch.matmul(x, w)`, `torch.relu(x)`, `torch.sum(x)` | Must be inside a `@T.prim_func` kernel |
| `torch.nn.functional` | `F.softmax(x, dim=-1)`, `F.linear(x, w)`, `F.relu(x)` | Must be inside a `@T.prim_func` kernel |
| tensor method compute | `x.sum()`, `x.mean()`, `x.softmax(dim=-1)`, `x.relu()` | Must be inside a `@T.prim_func` kernel |
| tensor operators | `x @ w`, `x + y`, `x * y`, `x / y` | Must be inside a `@T.prim_func` kernel |
| `nn.Module` calls | `self.conv(x)`, `self.linear(x)`, `self.layer(x)` | Must be inside a `@T.prim_func` kernel |

### Allowed in forward()

| Category | Examples | Purpose |
|----------|----------|---------|
| Kernel call | `func(a, b)` where `func` was compiled via `@tilelang.jit` | The TileLang kernel itself |
| Pinned memory ops | `x.contiguous()`, `x.npu()` | Ensure input is on NPU |
| Simple wrap | `return func(x, y)` | Return kernel output |

### Anti-Patterns (These Fail Conversion)

**1. Fully PyTorch — no kernel at all**
```python
# Forbidden: pure PyTorch, no TileLang kernel
def forward(self, x, w):
    return torch.matmul(x, w)
```

**2. Kernel defined but never used**
```python
@T.prim_func
def matmul_kernel(...):
    pass

class Model(nn.Module):
    def forward(self, x, w):
        return torch.matmul(x, w)  # Forbidden: kernel never called
```

**3. Mixed: partial kernel + partial torch**
```python
def forward(self, x, w):
    y = self.kernel_func(x, w)
    return y.sum(dim=-1)  # Forbidden: tensor method compute after kernel
```

**4. Tensor operators in forward**
```python
def forward(self, x, w):
    y = self.kernel_func(x, w)
    return y + 1  # Forbidden: + is a PyTorch operator
```

### Correct Pattern

```python
# Kernel compiled at module level
kernel = tile_operator(M, N, block_M, block_N)

class Model(nn.Module):
    def forward(self, x, y):
        return kernel(x, y)  # Allowed: call compiled TileLang kernel
```

## Quality Rules

- Keep the delivered output as a real TileLang NPU-backed implementation, not a pure PyTorch fallback. A pure PyTorch rewrite does not satisfy this convert task, even if differential tests pass.
- Do not introduce unnecessary code.
- Keep the converted file runnable as a PyTorch-facing operator artifact.
- Prefer targeted conversion over unrelated refactoring.
- Use the requested standalone or differential correctness validation mode instead of inventing a third validation workflow here.
- Input validation in the converted operator must limit itself to zero-cost metadata checks (`.dtype`, `.ndim`, `.device`, `.shape`, `.numel()`). Never scan tensor data for bounds or value-range validation — calling `.min().item()`, `.max().item()`, or any reduction+`.item()` on input tensors forces a GPU→CPU synchronization on every forward call and destroys performance. The caller is responsible for providing valid inputs, just as it is for the original PyTorch operator.
- Prefer Layer 1 base primitives for memory and compute (`T.alloc_shared`, `T.alloc_fragment`, `T.Parallel` + symbolic math). Use Layer 2 `T.tile.*` extensions when Layer 1 cannot express the needed operation. Layer 3 explicit hardware memory and manual sync are available but prefer the auto-managed approach — only use them when the kernel genuinely requires precise hardware-level control.

## When to Stop and Report

Do NOT replace broken TileLang kernels with PyTorch just to get validation green. A partial conversion that genuinely runs at least one TileLang kernel is valid; a conversion where `forward()` uses only PyTorch while TileLang kernels sit unused is not. If you cannot make a kernel correct after thorough debugging, report honestly to the user rather than hiding the problem.

**Litmus test**: if you deleted every `@T.prim_func` definition and `func = ...` call from the file, would `forward()` still produce a correct result? If yes, it is a fake conversion — the kernels are decoration, not the computation. Small PyTorch ops in `forward()` (dtype casts, shape manipulation, or ops TileLang cannot express like fp8 conversion) are acceptable as long as removing the TileLang kernels would break the output.

## Do Not

- Do not call `optimize` or create `opt-round-*` directories from this workflow.
- Do not create `baseline/` or any optimize-session artifacts from this workflow.
- Do not replace the converted TileLang kernel path with pure PyTorch just to get validation green.
- Do not create input-validation helpers (e.g., `_validate_index`, `_check_bounds`, `_assert_indices`, or similarly-named functions) that scan tensor data. Specifically, never call `.min().item()`, `.max().item()`, `.sum().item()`, or any reduction followed by `.item()` on GPU/NPU tensors before launching a kernel. These force a full-tensor GPU→CPU synchronization on every forward call. The converted operator inherits the same input contract as the original PyTorch operator — if the caller passes out-of-bounds indices, that is a caller bug, not something the conversion must guard against.
- Do not submit a pure PyTorch rewrite as the converted result, even when the wrapper signature or standalone or differential outputs still look correct.
- Do not write your own comparison script (e.g., `python3 -c "import torch; ..."`) to compare result `.pt` files or operator outputs.
- Do not use `torch.allclose`, `torch.testing.assert_close`, `torch.equal`, or any other numerical comparison function with custom tolerances to validate conversion correctness — always use `run-test-convert` from the `ascend-npu-run-eval` skill.
- Do not self-declare the conversion as "PASS" based on your own tolerance analysis — only the printed output of `run-test-convert` determines success.
- Do not run differential test files directly with `python3` — always use `cli.py run-test-convert`.
- Do not save custom `.pt` files (e.g., `REFERENCE_RESULT.pt`, `COMPARE_RESULT.pt`) for manual comparison — this may interfere with the CLI validation loop's result caching.
