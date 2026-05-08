# Attention Cube-Vector Pipeline Pattern

## Summary

Use this pattern for fused attention-style kernels that have two stages:

- a **matrix stage** that computes score tiles (for example via `tl.dot`)
- a **post-processing stage** that applies operations such as masking, scaling, normalization, optional stochastic transforms, and state writeback

In this card, "Cube" means the matrix stage and "Vector" means the post-processing stage.  
The goal is to reduce post-processing work and branch overhead while preserving numerical correctness and forward/backward agreement.

## Why This Pattern Exists

Across multiple optimization histories, two recurring experiences appeared:

1. **Early micro-tweaks often fail or give tiny gains.**  
   Simple launch-size or hint-only changes frequently plateau or regress when branch structure and dataflow are still mixed together.

2. **The successful path is usually staged, not one-shot.**  
   The best outcomes came from a sequence:
   - separate simpler and heavier semantic branches,
   - remove avoidable host-side data preparation for the simpler branch,
   - then tune only the true hotspot branch with bounded, evidence-driven thresholds/tiles.

This card encodes that staged strategy.

## Use When

- A regular matrix score path already exists, but post-processing dominates total latency.
- Mask conditions are recomputed repeatedly inside hot loops even though they depend only on host-known metadata (lengths, windows, causal mode).
- Scale and mask are applied as two separate passes over the same score tile.
- A simpler branch (optional feature disabled) is forced through the same heavy path as the feature-enabled branch.
- The code keeps extra state format conversions only to match a particular exponential implementation choice.
- Profile evidence shows the matrix stage is not the main limit; post-processing instruction count is.

## Avoid When

- The kernel is pure elementwise work with no meaningful matrix/post-processing split.
- The dominant bottleneck is launch or transfer overhead outside this stage.
- Correctness tolerance is extremely tight and cannot tolerate operation reordering or branch refactoring.

## Signals

### Code

- Repeated mask index arithmetic appears inside the normalization loop.
- Separate scale and mask passes both read/write the same score tensor.
- Simpler semantic branches still route through generic fallback code.
- Forward state uses one exponential-base convention while backward expects another.

### Profile

- Post-processing instruction count remains high after basic launch cleanup.
- The same post-processing kernels dominate across many shape regimes.
- Simpler branches still spend visible time in broadcast/cast preparation outside kernels.

## Repairs

### Recommended Order

Apply in this order unless evidence strongly suggests otherwise:

1. Branch/dataflow separation
2. Remove avoidable host-side preparation
3. Simplify repeated mask/scale work
4. Normalize state conventions
5. Hotspot-only bounded tuning
6. Optional architecture-gated refinements

This order reflects what repeatedly worked in practice.

### 1) Precompute and compact masks

If mask structure depends only on host metadata (sequence lengths, window limits, static causality), build compact masks once and feed kernels with ready-to-use layouts.

Use real valid extents in addressing shape, not padded maximum extents, when possible.

### 2) Fuse scale and mask at score boundary

Prefer a single expression feeding normalization instead of two separate passes:

```python
scores = scores * scale + tl.where(mask, 0.0, NEG_INF)
```

Choose `NEG_INF` according to dtype and numerical contract, and verify boundary behavior.

### 3) Split simpler and heavier branch dataflow

Treat the simpler semantic branch as a first-class fast path.  
Do not force it through heavier feature-enabled materialization if the math path is simpler.

### 4) Remove redundant host-side mask expansion

When mask tensors are naturally broadcastable, avoid eager host materialization (`expand` + `cast` + `contiguous`) if equivalent in-kernel broadcast/load is cheaper and still correct.

### 5) Keep exponential/state conventions consistent

If changing between equivalent formulations (for example `exp2`-based vs `exp`-based normalization), update saved state and backward formulas together. Never change forward state format in isolation.

### 6) Hotspot-only bounded tuning

After structural cleanup, tune only the branch and shape regime that clearly dominates runtime.  
Use bounded thresholds/tiles and stop when improvements become marginal or unstable.

### 7) Architecture-gated refinements only with proof

Device-specific compile choices can help, but must be guarded by verified target checks and regression-tested on non-target devices.

## Risks

- Mask precompute can silently change boundaries if shape assumptions drift.
- Scale+mask fusion can change NaN/Inf and conversion behavior.
- Branch splitting can duplicate logic and drift semantics between branches.
- State-format changes can break backward correctness even when forward looks fine.
- Hotspot tuning can overfit one regime and regress smaller cases.
- Architecture-specific options can accidentally leak into global defaults.

## What To Verify After Applying

- Correctness on variable-length, boundary, and heavy-mask cases.
- Forward/backward consistency, especially saved normalization state.
- Benchmarks split by branch regime (simpler vs heavier semantic branch).
- Operator mix confirms reduced broadcast/cast and repeated post-processing passes.
- Profile evidence shows lower post-processing instruction pressure on the targeted branch.
- After threshold/tile tuning, confirm small-shape guardrails still hold.

## Related Patterns

- `tiling`: when further gains depend on shape-specific tile ladders.
- `compile_hint`: for late-stage, low-risk lowering nudges after structural cleanup.
- `layout-store-and-block-pointers`: when output/store layout expression dominates cost.
