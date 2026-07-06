# Fuse Element-Wise Intermediate Computations into a Single Read-Once Kernel

## Summary

Replace a chain of PyTorch/ACL element-wise ops that all read from the same input tensors with a single Triton kernel that loads each input element once from global memory, computes every intermediate value in registers, and writes all outputs. Each incremental store added to the kernel costs less than a separate op launch plus its own full-memory-traffic pass. The net win compounds as more intermediates are fused: the kernel's memory pipeline is already saturated by its existing stores, so adding another store uses bandwidth that would otherwise be idle, while the eliminated PyTorch op saves both launch overhead and a full set of input loads.

## Use When

- An operator chain computes several element-wise intermediate tensors from the same set of input tensors via simple arithmetic (add, sub, mul, div).
- Profiling shows multiple PyTorch/ACL op launches dominating the timeline, each loading and writing full tensors independently.
- The intermediates are all elements of the same shape, so one kernel pass can cover all spatial positions without cross-element dependencies.
- The operation is not reduction-bound — the workflow is: compute element-wise intermediates, then reduce them to scalar statistics.
- For fp16 or bf16, precision-matching the reference requires care: Triton internal float32 accumulation may produce different rounding than step-by-step PyTorch ops at native precision.
- Every element-wise intermediate downstream of the shared inputs is fused into the kernel. Leaving even one intermediate as a standalone PyTorch op still incurs the launch overhead and full-tensor bandwidth cost of that op. The marginal cost of an extra store inside the already-running kernel is near zero; the cost of a separate PyTorch op is not.

## Signals

### Code

- A host-side function calls several PyTorch ops (`torch.add`, `torch.sub`, `torch.mul`, `torch.div`) on the same input tensors, each producing an intermediate tensor.
- The intermediates are then used in downstream reductions (`sum`, `mean`) or further element-wise ops.
- The shape is 4D or higher with batch, channel, and spatial dimensions, where a 2D grid (batch×channel, spatial) naturally handles per-channel stats.

### Profile

- Multiple op launches with similar duration appear sequentially in the trace, each touching the same input tensors.
- Memory-engine utilization is high across these ops because each independently loads the full input.

### IR

- Separate vector functions or op invocations for each intermediate, each with its own load path for the same inputs.

## Core Rewrite

Before — separate PyTorch ops, each loading inputs independently:

```python
x_centered = x - mean
x_normalized = x_centered / std
grad_output_scaled = grad_output * weight.view(1, C, 1, 1)
gos_xc = grad_output_scaled * x_centered
go_xn = grad_output * x_normalized
gos_over_std = grad_output_scaled / std
```

After — single Triton kernel, inputs loaded once per spatial position:

```python
@triton.autotune(configs=[...], key=["spatial_size"])
@triton.jit
def fused_intermediates_kernel(
    x_ptr, go_ptr, mean_ptr, weight_ptr, std_ptr,
    xc_ptr, gos_ptr, gos_xc_ptr, xn_ptr, go_xn_ptr, gos_over_std_ptr,
    batch, channels, spatial_size, bc_count,
    BC_PER_PROGRAM: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    RELOAD_XN: tl.constexpr,
):
    bc_pid = tl.program_id(0)
    spatial_pid = tl.program_id(1)
    spatial_offs = spatial_pid * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    spatial_mask = spatial_offs < spatial_size

    for i in range(BC_PER_PROGRAM):
        bc_idx = bc_pid * BC_PER_PROGRAM + i
        if bc_idx < bc_count:
            b = bc_idx // channels
            c = bc_idx % channels
            full_offs = b * channels * spatial_size + c * spatial_size + spatial_offs

            x_val = tl.load(x_ptr + full_offs, mask=spatial_mask, other=0.0)
            go_val = tl.load(go_ptr + full_offs, mask=spatial_mask, other=0.0)
            mean_val = tl.load(mean_ptr + b * channels + c)
            weight_val = tl.load(weight_ptr + c)
            std_val = tl.load(std_ptr + b * channels + c)

            xc = x_val - mean_val
            gos = go_val * weight_val
            gos_xc = gos * xc
            xn = xc / std_val
            gos_over_std = gos / std_val

            tl.store(xc_ptr + full_offs, xc, mask=spatial_mask)
            tl.store(gos_ptr + full_offs, gos, mask=spatial_mask)
            tl.store(gos_xc_ptr + full_offs, gos_xc, mask=spatial_mask)
            tl.store(gos_over_std_ptr + full_offs, gos_over_std, mask=spatial_mask)

            if RELOAD_XN:
                tl.store(xn_ptr + full_offs, xn, mask=spatial_mask)
                xn_reloaded = tl.load(xn_ptr + full_offs, mask=spatial_mask, other=0.0)
                go_xn = go_val * xn_reloaded
            else:
                go_xn = go_val * xn

            tl.store(go_xn_ptr + full_offs, go_xn, mask=spatial_mask)
```

Host side: allocate output tensors once, then reductions run over the stored intermediates:

```python
xc = torch.empty_like(x)
gos = torch.empty_like(x)
gos_xc = torch.empty_like(x)
xn = torch.empty_like(x)
go_xn = torch.empty_like(x)
gos_over_std = torch.empty_like(x)

grid = (bc_programs, triton.cdiv(spatial_size, BLOCK_SPATIAL))
fused_intermediates_kernel[grid](
    x, grad_output, mean, weight, std,
    xc, gos, gos_xc, xn, go_xn, gos_over_std,
    batch, channels, spatial_size, bc_count,
    BC_PER_PROGRAM=BC_PER_PROGRAM,
    RELOAD_XN=(grad_output.dtype != torch.float32),
)

# Reductions on stored intermediates
grad_weight = go_xn.sum(dim=(0, 2, 3))
grad_var = gos_xc.sum(dim=(2, 3), keepdim=True) * (-0.5) * torch.pow(std, -3)
```

## Precision: Store-Reload for fp16/bf16 Correctness

Triton accumulates in float32 registers by default. When computing chained products from register values, the result may differ from step-by-step PyTorch ops that round each operation to the native dtype before the next. Use store-reload to force dtype rounding: store the intermediate to global memory, then reload it. The reloaded value is rounded to the target dtype.

For fp32, skip the reload — register values are exact.

Gate with a `tl.constexpr` parameter:

```python
if RELOAD_XN:
    tl.store(intermediate_ptr + offs, val, mask=mask)
    val = tl.load(intermediate_ptr + offs, mask=mask, other=0.0)
# else: use register value directly
```

This store-reload applies ONLY when a chained product depends on a just-computed register value whose precision must match the reference. Simple ops (sub, mul of memory-loaded values) are bit-exact with aclNN on Ascend NPU and do not need store-reload. The constraint is: mul of a *just-computed* register value (still in float32) with another value must either use store-reload or accept the precision difference.

## BC Batching: Reduce Grid Dimension

When the batch×channel dimension is large, each program can handle multiple (b, c) pairs in a loop. This reduces grid size and launch overhead:

```python
if bc_count <= 32:    BC_PER_PROGRAM = 1
elif bc_count <= 128:  BC_PER_PROGRAM = 4
elif bc_count <= 512:  BC_PER_PROGRAM = 8
else:                  BC_PER_PROGRAM = 16
bc_programs = triton.cdiv(bc_count, BC_PER_PROGRAM)
```

## Shape-Gated Fallback

For extreme shapes where the fused kernel underperforms (e.g., many channels but tiny spatial dimension), fall back to native PyTorch ops. Gate the path selection on input shape at host dispatch time:

```python
if dtype == torch.bfloat16 and bc_count > 256 and spatial_size <= 16:
    # PyTorch native path
    ...
else:
    # Fused kernel path
    ...
```

## Avoid When

- The intermediates do not share the same input tensors — there is no redundant loading to eliminate.
- The element-wise ops are few (1-2) and the fused kernel overhead (grid launch, intermediate allocations) exceeds the savings.
- A reduction can be fused into the same pass for much greater savings (eliminating intermediate full-tensor stores entirely). Prefer reduction fusion over element-wise-only fusion when possible.
- The precision constraints cannot be met with store-reload, and the reference requires bit-exact matching.
- Partial fusion: fusing some element-wise intermediates into the kernel while leaving others as standalone PyTorch ops. If the leftover intermediate derives from the same set of loaded inputs (e.g. `gos/std` when `gos` is already computed in the kernel), it still pays a separate op launch and a full-tensor memory pass. Fuse the complete set of intermediates that derive from the shared inputs, or don't fuse at all.

## What To Verify After Applying

- Correctness matches reference for all supported dtypes (fp32, fp16, bf16), especially for chained products where store-reload rounds intermediate values.
- Geomean speedup improves; no individual case regresses beyond what's acceptable. If some shapes regress, add a shape-gated fallback.
- The kernel does not exceed UB budget. Each additional output adds a store buffer; verify the total allocation count fits within the compiler's buffer limit.
- For fp16/bf16, verify that chained products match the reference's step-by-step precision. If not, add store-reload at the right point in the chain.
- Audit the host-side code after the fused kernel: no element-wise arithmetic remains on the fused intermediates — all such work is inside the kernel. If a host-side op like `gos / std` remains on an intermediate the kernel already produced, that intermediate should be added as a kernel output instead.

## Related Patterns

- `sequential-kernel-fusion`: Fuses sequential Triton kernel launches. This pattern instead fuses PyTorch/ACL ops into a Triton kernel — the ops were never Triton kernels to begin with.
- `intra-kernel-pass-fusion`: Fuses two passes within a single row-wise kernel. This pattern co-locates multiple element-wise outputs within one spatial pass.
- `algebraic-optimization`: Reformulates the math to use fewer passes. This pattern keeps the same math but changes how the computations are dispatched.
- `program-multiple-rows`: Mapping multiple rows to one program_id. Used here via BC batching to reduce grid size.
- `autotune`: Tune BLOCK_SPATIAL based on spatial_size for optimal tile size.
- `kernelize-host-scatter-loops`: The opposite direction — kernelizes a host-side loop; this pattern replaces PyTorch op chains.
- `multi-output-kernel-writes`: A special case where the consumer op is an identity copy absorbed into the fused kernel.
