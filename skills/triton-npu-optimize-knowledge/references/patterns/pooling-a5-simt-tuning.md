---
id: pooling-a5-simt-tuning
priority: high
---

# A5 SIMT Spatial Pooling — Feature-Driven Tuning Playbook

## Summary

Methodology for **sliding-window spatial pooling** (any reduction: mean, max, etc.) on **A5 with `force_simt_only=True`**. Teaches **how to derive** dispatch, inner paths, and launch params from **shape / kernel / semantic features** — not fixed operator-specific constants. Each optimization point lists **source (why it applies), evidence (how to verify), and failure signature (when to revert)**.

## Use When

- Window reduction over NCDHW/NCHW layout with discrete per-output loads.
- A5 confirmed; kernel is scalar/index-heavy (see `a5-force-simt-only-discrete-access`).
- You need a **repeatable tuning order**: architecture → dispatch → inner path → block/warps.
- Harness has **many shapes**; accept/reject by **geomean**, not one case.

## Avoid When

- Not A5 SIMT, or Cube/matmul-dominated kernel.
- Mixing **SIMT slab** and **non-SIMT scalar row** kernels in the **same module/process** as fallback.

---

## 1. Derive case features (host, before coding)

For each JSON / bench case, compute:

| Feature | Formula / rule |
|---------|----------------|
| `out_w, out_h, out_d` | Standard pool output size (pad, stride, ceil, dilation) |
| `channels` (C) | Input C |
| `n_out` | `N × C × out_d × out_h × out_w` |
| `kernel_vol` | `kD × kH × kW` |
| `eff_span_w` | `dilation_w × (kW − 1) + 1` (and D/H) |
| `rows` (rowcol) | `N × C × out_d × out_h` |
| `pad_any` | any padding > 0 |
| `ceil` | ceil_mode |
| `dilated` | any dilation > 1 |
| `half` | fp16 / bf16 |
| `windows_inside` | last window fits in input (use **eff_span**, not raw k) |
| `reduction_cost` | mean: add+div; max: compare (+ optional index updates) |
| `semantic_flags` | e.g. count_include_pad, divisor_override, return_indices |

**Stack-depth proxy (inner loop weight):**

```
stack_pressure ≈ kernel_vol × (mask_branches + extra_state)
```

Increase when: non-full-window, clip bounds, CIP divisor, max+indices `tl.where` chain, half dtype, rowcol 2D tile.

Use these features to **route** — do not copy thresholds from another operator without re-benchmarking.

---

## 2. Tuning workflow (one variable class per round)

```
P0 Launch mode     → single SIMT launch exists?
P1 Outer dispatch  → flat vs rowcol (or flat-only)?
P2 Inner path      → full vs boundary (mask / clip / CIP)?
P3 Launch geometry → BLOCK_SIZE, num_warps, static unroll depth
P4 Reproducibility → fresh process, cache, full harness geomean
```

| Stage | Change | Pass evidence | Revert if |
|-------|--------|---------------|-----------|
| **P0** | `force_simt_only=True`, **one launch** per forward | Geomean jumps vs multi-launch / segmented grid | Correctness fail or no geomean gain |
| **P1** | flat ↔ rowcol routing | Target **feature bucket** improves (wide W, large vol) without geomean loss | Small shapes regress in geomean |
| **P2** | full ↔ mask ↔ clip inner | Pad/ceil/dilation semantics match reference | Half-pad or max+indices shapes regress sharply |
| **P3** | block / warps / unroll | 507035 gone; geomean ≥ prior | 507035, or geomean drops |
| **P4** | clean bench | Stable across two fresh runs | Single-case order-of-magnitude spike → co-JIT |

**Rules:** architecture before micro-params; full harness geomean each round; monolithic launch kernel + `constexpr` branches; `@triton.jit` **callees** from one launch kernel OK — **multiple launch entry points** for full/clip/mask not OK.

---

## 3. P1 — Derive flat vs rowcol

### When rowcol tends to help

**Source:** amortize 2D grid over many output rows; vectorize W with `start_w[None,:] + kw`.

**Favor rowcol when ALL hold:**

- `out_w` large enough that W-vector loads dominate flat 1D decode cost.
- `kernel_vol` large (many loads per output).
- Enough **row parallelism** (`rows = N×C×out_d×out_h` not tiny).
- Reduction is **cheap per tap** (mean accumulate) — not long compare+where+index chains.

**Evidence:** profile or bench — wide-W / large-kernel cases speed up; geomean not dragged down by small shapes.

### When flat tends to help

- Small `out_w`, low `channels`, moderate `kernel_vol` → 2D launch overhead > W-vector gain.
- Reduction carries **extra state** (max + indices, long `tl.where` chains) → rowcol often loses.
- Very large `n_out` but narrow W → flat 1D + large BLOCK often sufficient.

### How to set routing (derive, don’t copy)

Start with a **conservative flat default**. Add rowcol only for a **feature bucket** you can name, e.g.:

```text
rowcol if (out_w > W_wide) OR (kernel_vol > V_large AND channels > C_min)
```

**Tune `W_wide`, `V_large`, `C_min` by bisection on harness:**

1. Sort cases by `(out_w, kernel_vol, channels)`.
2. Enable rowcol for the worst flat bucket only; measure geomean.
3. Expand/shrink bucket until geomean peaks.

**Channel gate:** low-C + large-vol often stays flat — rowcol grid tax with few useful rows per tile.

**Reduction-type gate:** if max + return_indices, **default flat-only** until rowcol wins on geomean in isolation.

---

## 4. P2 — Derive inner path (full / mask / clip)

### USE_FULL_WINDOW (host `constexpr`)

**Predicate (derive per semantics):**

```text
full iff NOT pad_any AND NOT ceil AND NOT dilated
         AND windows_inside (eff_span per axis)
```

Kernel: fixed `kernel_vol` scan; no per-tap validity masks.

**Evidence:** unpadded, fully interior cases lose mask overhead; correctness unchanged.

### Boundary path selection

| Semantic need | Preferred inner | Source |
|-----------------|-----------------|--------|
| Include virtual pad in divisor (CIP) | Full kernel nest + **coordinate masks** + **one-shot divisor** from padded extent | Matches PyTorch CIP; avoid counting valid taps in loop |
| Exclude pad from divisor (closed) | **Clip-window** (`tstart/tend`, `d_len/h_len/w_len`) | See `pooling-clip-window-closed-divisor` |
| Dilation > 1 | **Coordinate masks** only (no clip shortcut) | Clip assumes unit stride in kernel index space |
| Max + half + pad | **Coordinate masks** over clip-window | Clip nest + compare+where heavier on SIMT stack |
| Max + indices | Masks + **fp32 compare** + optional linear index | Numeric stability; index tied to masked loads |

**CIP divisor (once per lane, before load nest):**

```python
cip_axis = max(min(start + K, in + PAD) - start, 0)
divisor = cip_d * cip_h * cip_w   # fp32 for mean
```

**Empty window:** detect `tstart >= tend` or zero lengths → output 0; clamp divisor to avoid div-by-zero.

**How to choose between mask vs clip for the same dilation=1 case:**

1. Implement the semantically correct path first.
2. If **507035** or half-pad geomean regression → try the other path **only if semantics allow**.
3. Keep the path with **better geomean** at equal correctness — do not assume clip is always faster.

---

## 5. P3 — Derive BLOCK_SIZE and num_warps

### Flat 1D grid

**BLOCK_SIZE** scales with **`n_out`** (more outputs → larger blocks to amortize launch):

```text
if use_full_window:
    tier by n_out: large → 2048, medium → 1024, default → 512
else:
    # stack_pressure high → cap block (often 512)
    if half OR deep_inner: prefer 512
    elif n_out large: may use 1024 if no 507035
```

**num_warps:**

```text
default 8 for full-window / low stack_pressure
if stack_pressure high (non-full, clip, half, deep nest):
    try 4 warps
```

**Evidence for reduction:** 507035 disappears; geomean stable or up.

### Rowcol 2D grid

**W tile (`block_size`):**

- Ascend: **`tl.arange(0, BLOCK)` needs pow2 BLOCK** → `block_size = next_pow2(out_w)` capped by a max tile.
- Larger `out_w` → wider tile (up to hardware-friendly cap: 64 / 128 / 256 tiers).

**Row tile (`block_rows`):**

- Scale with `rows`: more rows → more `block_rows` (tier: 4 / 8 / 16 / 32).
- Very large `kernel_vol` → favor smaller `block_rows` if stack_pressure high.
- **Non-full-window:** cap `block_rows` (e.g. `min(computed, 8)`) — evidence: 507035 on pad paths.

Derive tiers by benching **one representative case per (out_w tier, kernel_vol tier, full vs pad)** then fix host picker.

### Static unroll (W axis only)

**Source:** small fixed kW reduces loop control overhead.

```text
if kW in {2,3,7}: tl.static_range(kW)
else: range(kW)
```

**Never** static-unroll full D×H×W when `kernel_vol` is large — **507035**.

Adding `static_range(kW)` for new k: trial on **max kernel_vol case** in harness; revert on 507035 or geomean loss.

### fp32 intermediate

**Source:** numeric stability for mean sum/divide and max compare.

Load → `.to(tl.float32)` for accumulate/compare; store in native dtype.

**Evidence:** correctness on half; geomean usually neutral vs native half acc — keep fp32 unless profile shows convert-bound and correctness allows otherwise.

---

## 6. Reduction-type feature matrix (re-derive per op)

Do **not** assume routing for one reduction applies to another. Re-run P1–P3.

| Feature | Mean-like (sum ÷ divisor) | Max-like (max, optional arg index) |
|---------|---------------------------|-------------------------------------|
| Rowcol benefit | Often on wide W + large vol + sufficient C | Often **low**; compare+where+index amplifies rowcol cost |
| Non-full warps | Usually lower warps whenever not full-window | Lower warps when **half + non-full**; fp32 may keep 8 |
| Non-full block | Usually cap 512 | Cap 512 when half; else tier by `n_out` |
| Inner path | CIP → mask + one-shot divisor; closed → clip | Prefer mask over clip on half pad; dilation → mask only |
| Extra state | divisor_override, empty→0 | sentinel init, fp32 compare, return_indices |
| Fallback to scalar row kernel | Risky — validate per module | **High failure risk** same module/process as SIMT |

---

## 7. Optimization points — source, evidence, signature

| Point | Source (why) | How to verify | Failure signature |
|-------|--------------|---------------|-------------------|
| Single SIMT launch | Launch overhead dominated multi-grid designs | One kernel in trace; geomean up | Still low geomean → inner path bound, not launch |
| `force_simt_only=True` | A5 discrete access path | Profile scalar ratio; geomean vs default | No gain → not discrete-dominated |
| USE_FULL_WINDOW | Interior outputs need no masks | Unpad cases faster; correct | No speedup → already memory bound |
| Flat/rowcol split | Match parallelism to shape | Bucket geomean improves | Small-shape regression → shrink rowcol bucket |
| CIP one-shot divisor | Avoid per-tap counting | Correct pad semantics; geomean vs clip-acc | Wrong output → divisor formula |
| Coordinate mask vs clip | Stack depth vs mask count tradeoff | A/B on pad half cases | 507035 or geomean drop → switch path |
| Pow2 W tile (rowcol) | Ascend arange constraint | Correctness + rowcol cases speed up | Illegal block or slow → fix pow2 |
| Lower warps / block | SIMT stack limit | 507035 cleared | Slower than prior → partial revert |
| Monolithic kernel | co-JIT from multiple launch kernels | Stable bench in fresh process | One case ×10 latency → split launch pollution |
| No SIMT+row mix | Different codegen paths interfere | Never import/embed scalar row in SIMT module | Geomean collapse across many cases |

---

## 8. Anti-patterns (feature-triggered, not operator-specific)

| Trigger feature | Anti-pattern | Why |
|-----------------|--------------|-----|
| A5 SIMT pooling | W-slab / gather slab | Measured loss vs flat/rowcol discrete loads |
| Any | Global interior/boundary **multi-launch** | Launch tax |
| Small vol + low C | Force rowcol | Grid overhead |
| Any | In-kernel `tl.where(interior, fast, slow)` | Branch + reg pressure |
| Any | Separate **launch** kernels per full/clip/mask | co-JIT / compile interference |
| Large kernel_vol | Full D×H×W static unroll | 507035 |
| SIMT optimize module | Scalar row fallback (embed or import) | Path degeneration |
| Max-like + indices | Rowcol without geomean proof | where-chain × W tile |
| Bench session | Mixed kernel architectures same process | JIT cache pollution → phantom spikes |

**Neutral (try once, expect no geomean win):** global half accumulate; clip+accum inline merge when load-bound.

---

## 9. Diagnostics (relative, not absolute geomean targets)

- **Geomean step-change** after P0 → launch was the bottleneck.
- **Geomean drop** after dispatch change → bucket too wide; narrow rowcol predicate.
- **Geomean drop** after inner-path swap → wrong path for that semantic bucket; revert.
- **507035** → reduce stack_pressure (warps↓, block↓, unroll↓).
- **One case spike, others OK** → fresh Python process + clear Triton cache; suspect co-JIT.
- **Correctness ok, geomean flat** → load/memory bound; dispatch tweaks won't help much.

---

## 10. Acceptance checklist

- [ ] Features computed for all harness cases; routing explainable per bucket.
- [ ] Semantics: dtypes, pad/ceil, CIP/closed divisor, dilation, indices if any.
- [ ] One launch per forward.
- [ ] Full harness geomean vs chosen baseline; two stable fresh-process runs.
- [ ] No §8 anti-patterns.
- [ ] If reduction type or extra semantics change → re-run P1–P3.

---

## Related Patterns

- `a5-force-simt-only-discrete-access` — when to enable SIMT launch
- `pooling-clip-window-closed-divisor` — closed-divisor clip inner loop
- `pooling-inner-w-slab-gather` — not for A5 SIMT pooling
- `flat-index-decode-tiling` — flat outer tiling
