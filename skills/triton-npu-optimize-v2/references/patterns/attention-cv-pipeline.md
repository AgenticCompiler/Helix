# Attention Cube-Vector Pipeline Pattern

## Summary

Reduce latency in Cube+Vector fused attention-like kernels by cutting vector-side instruction pressure, making mask/scale work cheaper, and using architecture-gated compile options only when the target device supports them.

Use this after the kernel is already structurally sound. These optimizations are sensitive to numerics, architecture, and forward/backward consistency.

## Use When

- A `tl.dot` loop is followed by substantial vector epilogue work such as scale, mask, softmax, dropout, or bias.
- Profiling suggests Cube and Vector work are close enough that vector-side overhead limits overlap.
- A loop repeatedly recomputes the same mask tensor from sequence lengths or causal indices.
- Scale and mask are separate operations before softmax.
- The code stores log-sum-exp state in a base-2 representation solely because the forward path uses `exp2`.
- The target is known to be an A5 device such as `ascend950PR` or `ascend950DT`.

## Avoid When

- The kernel is pure Vector work rather than Cube-plus-Vector fused work.
- Profiling shows memory transfer, not vector epilogue work, is the dominant bottleneck.
- Architecture-specific compile settings cannot be gated on verified target information.

## Signals

### Code

- A `tl.dot` loop is followed by repeated mask, scale, softmax, dropout, or bias work on the vector side.
- The same mask tensor is recomputed inside a hot loop even though it depends only on host-known metadata.
- The forward path stores log-sum-exp state in base-2 form solely because it uses `exp2`.

### Profile

- Profiling suggests Cube and Vector work are close enough that vector-side instruction pressure is limiting overlap.
- The kernel is structurally sound, but the post-dot vector path still appears to dominate latency.

## Repairs

### Cube/Vector pipeline scheduling

Move independent vector work away from the critical Cube path so loads, `tl.dot`, and epilogue work can overlap better. Prefer changes that reduce live vector temporaries and instruction count before adding buffering.

Do not use this pattern when the kernel is pure Vector or when profiling shows memory transfer, not vector epilogue work, is dominant.

### Precompute repeated masks

If mask construction is repeated inside a hot loop and depends only on host-known metadata, precompute the mask on the host and pass it as a tensor. In varlen cases, build each batch mask so invalid positions are already false, then use block pointer shapes that reflect the real `(q_len, k_len)` region.

This trades memory bandwidth for less vector control work. Validate the tradeoff with benchmark evidence.

### Fuse scale and mask

When softmax scores are scaled then masked, consider combining the operations into one expression that feeds softmax directly:

```python
scores = scores * scale + tl.where(mask, 0.0, -float("inf"))
```

Use a finite large negative value only when dtype and downstream numerics make that equivalent and safe.

### Use `exp` instead of `exp2` consistently

If `exp2(x * log2e)` is used only to approximate `exp(x)`, consider switching to `tl.exp(x)` and store matching log-sum-exp state. Update backward formulas together; forward-only changes can silently break gradients.

### Architecture-gated compile parameters

Some compile parameters are only appropriate for A5 targets. Gate them on the actual architecture and record the target evidence. Do not enable A5-only options on older Ascend devices.

## Risks

- Mask precomputation can change boundary behavior if block pointer shapes describe the padded max shape instead of real lengths.
- Scale-mask fusion changes where infinities and dtype conversions happen.
- `exp` versus `exp2` must be consistent across saved state and backward code.
- A5 compile flags are target-specific and must not become unconditional defaults.

## What To Verify After Applying

- Run correctness with mask-heavy, boundary, and varlen cases when available.
- Compare both latency and profiler balance; a vector-instruction reduction should show up in the evidence.
- Record any architecture gate in `attempts.md` and `summary.md`.
- If backward code exists, verify forward/backward state conventions together.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Batch-2 track for **`15_AttentionSoftmaxWithSoftcappingAndDropout`** (vector/Cube epilogue side; `workspace/NPUKernelBench_level_1_2_triton/15_AttentionSoftmaxWithSoftcappingAndDropout/`). Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-7` (parent `opt-round-6`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** Causal mask predicates were fused into the inner softmax loop as repeated `tl.where` chains on logits.
- **Change:** Precomputed a compact mask tensor (or mask band) from host-known `seqlen` / window once per tile, then consumed it in softmax without recomputing indices.
- **Evidence:** Vector instruction count drop in profiler excerpt; `summary.md` varlen mask-heavy case.
- **Interpretation:** Matches “mask recomputed inside hot loop” anti-pattern from this card’s signals.

**`opt-round-9` (parent `opt-round-8`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Scale and mask were separate elementwise passes before softmax, doubling bandwidth on logits.
- **Change:** Fused scale+mask into a single vector op feeding the max-shifted softmax stable path.
- **Evidence:** `attempts.md` fused expr; `summary.md` dense attention case.
- **Interpretation:** Demonstrates “scale and mask separate” repair from `## Use When`.

**`opt-round-10` (parent `opt-round-9`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** Forward stored log-sum-exp in base-2 only because `exp2` was used in softmax; vector epilogue paid extra conversions.
- **Change:** Switched to `tl.exp` consistently and aligned saved state representation with backward expectations (documented in round notes).
- **Evidence:** Numerics checklist in `attempts.md`; paired forward/back tests; `summary.md` softmax path.
- **Interpretation:** Applies the card’s `exp` vs `exp2` consistency rule—do not change without backward audit.

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators in this excerpt: **`25_MaskedSoftmaxWithAttentionDropoutBackward`** (others in batch 5 map to **`tiling.md`**, **`program-multiple-rows.md`**, etc.). Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `25_MaskedSoftmaxWithAttentionDropoutBackward`

**`opt-round-5` (parent `opt-round-3`)** — `25_MaskedSoftmaxWithAttentionDropoutBackward/opt-round-5/summary.md`

- **Kernel / round / parent:** `25_MaskedSoftmaxWithAttentionDropoutBackward` / `opt-round-5` / `opt-round-3`.
- **Pre-change scenario:** Cases **1–3** (`p_dropout == 0`) still showed **host-only** softmax-backward ops with **no matched Triton kernel** while r3 improved the dropout branch (`summary.md`).
- **Change:** Added **`_masked_softmax_backward_nodropout_kernel`** + wrapper routing; **preserved** r3 **`_dropout_backward_scale_kernel`** for dropout-on cases.
- **Evidence:** Correctness passed; `compare-perf` **Avg +23.7%**, **1.38×** geomean, **1.22×** total vs baseline; large wins cases **1–3** and smaller wins **4–5** (`summary.md`).
- **Interpretation:** Attention-backward sessions must **split branches** by **`p_dropout`**—vector epilogues should not sit entirely in host wrappers.

**`opt-round-6` (parent `opt-round-5`)** — `25_MaskedSoftmaxWithAttentionDropoutBackward/opt-round-6/attempts.md` + `opt-note.md`

- **Kernel / round / parent:** `25_MaskedSoftmaxWithAttentionDropoutBackward` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** No-dropout path still paid **`BroadcastTo` + `Cast`** from **`expand(...).contiguous().to(int8)`** host mask materialization (`attempts.md`).
- **Change:** Load **original attention mask in-kernel**; broadcast head dim when **`mask.shape[1] == 1`**; drop expanded **`int8`** buffer on that path.
- **Evidence:** Correctness passed; further gains on no-dropout cases while dropout cases still improve (`opt-note.md`); **promoted** toward r7–r10 block ladder.
- **Interpretation:** Same card theme as forward mask work—**avoid redundant mask expansion** on the hot vector path.

**`opt-round-3` (parent `baseline`)** — theme in `25_MaskedSoftmaxWithAttentionDropoutBackward/opt-note.md`

- **Kernel / round / parent:** `25_MaskedSoftmaxWithAttentionDropoutBackward` / `opt-round-3` / baseline.
- **Pre-change scenario:** Flat launch + heavy dropout mask application on hot path (`opt-note.md`).
- **Change:** **Simplified dropout mask application** while keeping flat launch structure that r2 had destabilized.
- **Evidence:** Correctness passed; small **net win** vs baseline; case **4** improved (`opt-note.md`); **promoted** parent for r5 split.
- **Interpretation:** First stable vector-epilogue cleanup before branch split and width-tier ladder (**`tiling.md`** rounds 7–10).
