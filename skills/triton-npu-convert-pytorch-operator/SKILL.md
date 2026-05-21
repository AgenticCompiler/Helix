---
name: triton-npu-convert-pytorch-operator
description: Convert one PyTorch operator into a Triton NPU-backed PyTorch operator, preserve the trailing input-helper block, and validate the converted output through differential testing.
---

# Convert PyTorch Operator

Convert one PyTorch operator file into a PyTorch-facing operator backed by a real Triton Ascend NPU kernel path.

Use this skill when the user wants a new converted operator artifact instead of an in-place optimize round.

## Inputs

- one original PyTorch operator file
- one requested output path for the converted operator
- one requested differential test mode
- optional remote execution context from the outer task

## Outputs

- one converted operator file, usually named `triton_<origin-name>.py`
- preserved trailing input-helper block in the converted output
- one generated differential test file for the converted output
- a short summary of what was converted, what remained unchanged, and any blockers

## Required Workflow

1. Read the original operator file carefully before editing anything.
2. Treat the original input operator file as source material and correctness oracle.
3. Do not execute the original input operator file.
4. Identify the public PyTorch entrypoint that should remain visible after conversion.
5. Convert the operator so the delivered output remains PyTorch-facing but implements the converted computation through a real Triton Ascend NPU kernel path.
6. A PyTorch-facing wrapper or `torch.nn.Module` public API may remain when that is the intended interface, but the converted computation itself must stay on the Triton kernel path.
7. You may replace some operators, leave some unchanged, fuse operations, or make algorithmic changes when that helps the Triton NPU path.
8. Write the converted operator to the requested output path instead of overwriting the original file.
9. Preserve the trailing input-helper block from the input file in the converted output because later harness generation and validation may need it.
10. In the common contract shape, preserve helpers such as `get_init_inputs()` and `get_inputs()` instead of dropping or rewriting them away.
11. Do not introduce unnecessary wrappers, compatibility branches, helper layers, or scaffolding that do not materially serve the converted Triton NPU path.
12. Target Ascend NPU only for this conversion flow; do not add CUDA, CPU, MPS, or generic multi-backend fallback logic unless the source file already requires shared import structure around the public API.
13. Do not add differential test code directly into the converted operator file.
14. Use `triton-npu-gen-test` to generate a differential test for the converted output.
15. Use the original input operator as the differential reference implementation and the converted output as the system under test.
16. Use `triton-npu-run-eval` to execute the differential test against the converted output.
17. If the converted output hits Triton compile, JIT, launch, or kernel-structure errors, use `triton-npu-repair-guide` for operator-side repair heuristics.
18. Finish only after the differential test passes or a clear environment blocker prevents further progress.

## Converted Example

Use a real converted output example in the generated file, not only a prose description. For a simple elementwise add conversion, a converted output may look like this:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=128)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return triton_add(a, b)


N = 2048


def get_inputs():
    a = torch.randn(N, N, device="npu")
    b = torch.randn(N, N, device="npu")
    return [a, b]


def get_init_inputs():
    return []
```

In this kind of conversion:

- `def triton_add(...)` is the PyTorch-facing wrapper that calls the Triton kernel
- `class ModelNew` is the converted public architecture
- the trailing `get_init_inputs()` / `get_inputs()` block is preserved in the converted output instead of being dropped
- the original source operator remains the correctness oracle for differential validation

## Forward Method Constraints

The converted operator **must** be a pure Triton Ascend implementation. The `forward()` method may only allocate buffers and launch kernels — all computation must happen inside `@triton.jit` kernels.

### Forbidden in forward()

| Category | Examples | Reason |
|----------|----------|--------|
| `torch` compute functions | `torch.matmul(x, w)`, `torch.relu(x)`, `torch.sum(x)` | Must be inside a `@triton.jit` kernel |
| `torch.nn.functional` | `F.softmax(x, dim=-1)`, `F.linear(x, w)`, `F.relu(x)` | Must be inside a `@triton.jit` kernel |
| tensor method compute | `x.sum()`, `x.mean()`, `x.softmax(dim=-1)`, `x.relu()` | Must be inside a `@triton.jit` kernel |
| tensor operators | `x @ w`, `x + y`, `x * y`, `x / y` | Must be inside a `@triton.jit` kernel |
| `nn.Module` calls | `self.conv(x)`, `self.linear(x)`, `self.layer(x)` | Must be inside a `@triton.jit` kernel |

### Allowed in forward()

| Category | Examples | Purpose |
|----------|----------|---------|
| buffer alloc | `torch.empty(shape)`, `torch.zeros(shape)`, `torch.ones(shape)` | Allocate output for kernel |
| shape ops | `x.view(...)`, `x.reshape(...)`, `x.permute(...)`, `x.transpose(...)` | No compute involved |
| metadata | `x.shape`, `x.dtype`, `x.device`, `x.numel()` | Needed for grid calculation |
| kernel launch | `kernel[grid](...args)` | Call a custom `@triton.jit` kernel |

### Anti-Patterns (These Fail Conversion)

**1. Fully PyTorch — no kernel at all**
```python
# ❌ FAILS: pure PyTorch, no Triton kernel
def forward(self, x, w):
    return torch.matmul(x, w)
```

**2. Kernel defined but never called**
```python
@triton.jit
def matmul_kernel(...):
    pass

def forward(self, x, w):
    return torch.matmul(x, w)  # kernel defined but unused
```

**3. Mixed: partial kernel + partial torch**
```python
def forward(self, x, w):
    y = self.kernel[grid](x, w)
    return y.sum(dim=-1)  # ❌ tensor method compute after kernel
```

**4. Tensor operators in forward**
```python
def forward(self, x, w):
    y = self.kernel[grid](x, w)
    return y + 1  # ❌ + is a PyTorch operator
```

### Correct Pattern

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx)
    y = tl.load(y_ptr + idx)
    output = x + y  # ✅ compute inside kernel
    tl.store(output_ptr + idx, output)

class ModelNew(nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)  # ✅ allowed: buffer alloc
        add_kernel[(1,)](x, y, output, x.numel(), BLOCK_SIZE=128)  # ✅ allowed: kernel launch
        return output  # ✅ allowed: return kernel output
```

## Quality Rules

- Keep the original input operator file unchanged.
- Keep the delivered output as a real Triton NPU-backed implementation, not a pure PyTorch fallback.
- A pure PyTorch rewrite does not satisfy this convert task, even if differential tests pass.
- Do not introduce unnecessary code.
- Keep the implementation focused on the Ascend NPU path instead of adding generic backend handling.
- Keep the converted computation on a real Triton Ascend NPU kernel path even when the public API stays PyTorch-facing.
- Preserve the input file's trailing input-helper block.
- Keep the converted file runnable as a PyTorch-facing operator artifact.
- Prefer targeted conversion over unrelated refactoring.
- Use differential correctness validation instead of inventing a second validation workflow here.
- Do not call tensor reduction ops (`.min()`, `.max()`, `.sum()`, `.mean()`, etc.) followed by `.item()` on GPU/NPU input tensors in the kernel-launch path. This pattern forces a GPU→CPU synchronization and scans entire tensors, defeating the performance purpose of the conversion. Metadata checks (`.dtype`, `.ndim`, `.device`, `.shape`, `.numel()`) are safe and do not cause synchronization.

## Do Not

- Do not execute the original input operator file.
- Do not overwrite the original input operator file.
- Do not drop the trailing `get_init_inputs()` / `get_inputs()` helper block when it exists.
- Do not add CUDA-only, CPU-only, MPS, or generic multi-backend dispatch branches when the converted kernel is meant for Ascend NPU.
- Do not add defensive backend-selection code that is unnecessary for this Ascend NPU conversion workflow.
- Do not call `optimize` or create `opt-round-*` directories from this workflow.
- Do not create `baseline/` or any optimize-session artifacts from this workflow.
- Do not replace the converted Triton kernel path with pure PyTorch just to get validation green.
- Do not call `.item()` on a GPU/NPU tensor that is the result of a reduction op (`.min()`, `.max()`, `.sum()`, `.mean()`, etc.) in the kernel-launch path — this forces a device→host synchronization that scans the full tensor. Tensor metadata checks (`.dtype`, `.ndim`, `.device`, `.shape`, `.numel()`) and `.item()` in non-hot-path code (initialization, test data generation) are fine.
- Do not submit a pure PyTorch rewrite as the converted result, even when the wrapper signature or differential outputs still look correct.
