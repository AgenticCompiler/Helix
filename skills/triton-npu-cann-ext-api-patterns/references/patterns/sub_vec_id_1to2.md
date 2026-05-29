# 1:2 `sub_vec_id` Rewrite Pattern

## Goal

Split work so **vector units** handle **half** of the chosen tile axis (`sub_vec_id` lanes 0/1), while **cube** keeps full-tile `tl.dot` math. Use explicit staging for vector<->cube handoff (prefer `fixpipe` for cube->vector where applicable), and preserve numerical equivalence with the vanilla kernel.

## Why 1:2 rewriting helps

On Ascend, mixed vector+cube kernels can hit UB/CBUF/register pressure before reaching larger tile sizes. A 1:2 split often reduces vector-path pressure while keeping cube utilization on full tiles. In practice this can:

- unlock larger compile-time tiles that vanilla kernels cannot sustain,
- improve occupancy/throughput on vector-heavy tails,
- reduce unnecessary staging when combined with fixpipe-based cube->vector handoff.

## Principles (must hold)

1. **Vector does vector work; cube does `tl.dot`** — keep non-dot operations in vector path; keep full-tile reductions in cube path.
2. **Both lanes do real work** — lane ownership must be disjoint and symmetric; no lane-0 computational fallback.
3. **Handoff is explicit and synchronized** — when lanes co-produce a full tile, stage into shared buffer and synchronize before full-tile cube loads.
4. **No semantic shortcuts without proof** — do not replace full-dot semantics with partial-dot approximations unless validated mathematically and numerically.
5. **Reviewability first** — keep `_sub_vec` changes structure-preserving: aligned function order, minimal unrelated refactors, clear before/after paths.

## Quick Start

1. **Copy, don't replace**: duplicate the vanilla kernel module to `_sub_vec` and keep vanilla unchanged.
2. **Split vector work by lane**: use `vec_id = tle.sub_vec_id()` and half-offset indexing for disjoint lane ownership.
3. **Keep cube math full**: preserve full-tile `tl.dot` semantics; do not shrink `tl.dot` unless mathematically proven.
4. **Stage handoff explicitly**: for vector->cube, write lane halves to scratch, then `tl.debug_barrier()`, then load full tile for cube.
5. **Prefer fixpipe for cube->vector**: use `tle.fixpipe(..., NZ2ND, ROW_SPLIT)` when applicable.
6. **No fake fallback**: don't route real computation through lane 0 only or silently call vanilla in `_sub_vec`.
7. **Validate and measure**: run accuracy against the same golden as baseline, then benchmark and record before/after with tile configs.
8. **Keep diffs reviewable**: preserve function order and signatures; avoid unrelated cleanups.

## Lane Setup

```python
import triton.language.extra.cann.extension as tle

HALF_BT: tl.constexpr = BT // 2
vec_id = tle.sub_vec_id()           # expected 0 or 1 in CV-fusion kernels
half_offset = (vec_id * HALF_BT).to(tl.int32)
```

Use `half_offset` in `make_block_ptr` origins so lanes cover disjoint halves.

## Mandatory rules

1. **No fallback masking of lane work** — `vec_id == 0` guards are only acceptable for write-once shared outputs to avoid races.
2. **No fake sub-vec path** — do not keep `_sub_vec` entry points that simply call vanilla kernels.
3. **Full-tile cube math** — keep `tl.dot` operands full where required by original math.
4. **Explicit synchronization** — if two lanes fill one full tile in shared scratch, use `tl.debug_barrier()` before full-tile load for cube.
5. **`make_block_ptr` parity with vanilla** — if vanilla uses `tl.make_block_ptr` for a region, keep that style in `_sub_vec`. Do not replace those vanilla `make_block_ptr` regions with raw pointer arithmetic during rewrite.
6. **Preserve 1:2 logic while restoring pointers** — keep half-tile pointers (`HALF_BT`, `row_start`/`half_offset`) so lane 0 and lane 1 always write disjoint regions.

## Data handoff strategies

### Vector -> cube

When cube needs full tiles assembled from lane halves:

- add explicit scratch pointer args
- each lane stores its half into disjoint slice
- `tl.debug_barrier()`
- cube loads full tile and runs `tl.dot`

### Cube -> vector

Prefer Ascend fixpipe when supported:

```python
b_half = tle.fixpipe(
    b_cube,
    dma_mode=tle.FixpipeDMAMode.NZ2ND,
    dual_dst_mode=tle.FixpipeDualDstMode.ROW_SPLIT,
)
```

If fixpipe is not applicable, use full-tile store + half-tile load + barrier.

### `al.scope` / L1 handoff

If using `al.scope` and L1 staging instead of global scratch:

- use explicit `al.sync_block_set` / `al.sync_block_wait` around producer/consumer boundaries
- do not rely on entering/leaving `al.scope` for synchronization
- avoid "shape/dtype update" assumptions across scope boundaries; use fresh temporaries

Fractal subblock sizing for staged cube operands:

- fp16 input to `tl.dot`: 16×16
- fp32 input to `tl.dot`: 16×8

## Autotune and tile policy

1. Sweep bounded tile sets first: `BT in {64, 128, 192, 256}`, `BK=BV in {32, 64, 128, 256, 512}`.
2. Keep `num_warps` / `num_stages` fixed initially.
3. Do not pass tuned meta-params from both Python call and `triton.Config`.
4. Use `grid(meta)` based on tuned meta values.
5. Use `TRITON_PRINT_AUTOTUNING=1` to capture winners.

```python
@triton.autotune(
    configs=[
        triton.Config({"BT": 128, "BK": 64, "BV": 64}),
        triton.Config({"BT": 192, "BK": 64, "BV": 64}),
    ],
    key=["H", "K", "T"],
)
@triton.jit
def k(..., BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    ...

def grid(meta):
    return (triton.cdiv(T, meta["BT"]), triton.cdiv(V, meta["BV"]), B * H)

k[grid](..., T=T, H=H, K=K, V=V)  # BT/BK/BV only from Config/meta
```

## Experiment protocol

For each kernel:

1. **Baseline**: record current timing per benchmark category.
2. **Port**: implement 1:2 in `_sub_vec` copy.
3. **Optimize**: remove redundant `temp_*`, prefer fixpipe where valid.
4. **Tune**: run bounded autotune and capture best configs.
5. **Validate**: numerical checks vs same golden reference as baseline.
6. **Report**: before/after table with tiles and timing.

## Duplication and rollout workflow

- keep vanilla source unchanged
- implement in sibling `_sub_vec` source file
- duplicate harness/driver to `_sub_vec`
- duplicate acc/perf tests to `_sub_vec`

Ensure `_sub_vec` harness/tests import `_sub_vec` symbols only.

## Pattern recipes

### Pattern A: half stores -> full dot

```python
HALF_BT: tl.constexpr = BT // 2
vec_id = tle.sub_vec_id().to(tl.int32)
half_off = (vec_id * HALF_BT).to(tl.int32)

p_k_half = tl.make_block_ptr(
    k_ptr, (K, T), (1, H * K),
    ((i_k * BK).to(tl.int32), (i_t * BT + half_off).to(tl.int32)),
    (BK, HALF_BT), (0, 1),
)
b_k_half = tl.load(p_k_half, boundary_check=(0, 1))

p_tmp_half = tl.make_block_ptr(
    temp_k + scratch_base, (BK, BT), (BT, 1),
    (0, half_off), (BK, HALF_BT), (1, 0),
)
tl.store(p_tmp_half, b_k_half, boundary_check=(0, 1))
tl.debug_barrier()

p_tmp_full = tl.make_block_ptr(
    temp_k + scratch_base, (BK, BT), (BT, 1),
    (0, 0), (BK, BT), (1, 0),
)
b_k_full = tl.load(p_tmp_full, boundary_check=(0, 1))
b_out += tl.dot(b_k_full, b_v_full)
```

### Pattern B: full dot -> half vector postprocess

```python
HALF_N: tl.constexpr = BLOCK_N // 2
sub_id = tle.sub_vec_id().to(tl.int32)
offs_n_half = pid_n * BLOCK_N + sub_id * HALF_N + tl.arange(0, HALF_N)

acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k0 in range(0, K, BLOCK_K):
    a = tl.load(...)
    b = tl.load(...)
    acc += tl.dot(a, b)
tl.store(temp_full_ptrs, acc, mask=full_mask)

t = tl.load(temp_half_ptrs, mask=half_mask, other=0.0).to(tl.float32)
bias = tl.load(bias_half_ptrs, mask=half_mask, other=0.0)
tl.store(out_half_ptrs, (t + bias).to(out_dtype), mask=half_mask)
```

### Pattern C: cube/vector/cube pipeline

```python
b_o_cube = tl.dot(b_q_full, b_h_full)
b_A_cube = tl.dot(b_q_full, b_k_full)

b_o_half = tle.fixpipe(b_o_cube, dma_mode=tle.FixpipeDMAMode.NZ2ND,
                        dual_dst_mode=tle.FixpipeDualDstMode.ROW_SPLIT)
b_A_half = tle.fixpipe(b_A_cube, dma_mode=tle.FixpipeDMAMode.NZ2ND,
                        dual_dst_mode=tle.FixpipeDualDstMode.ROW_SPLIT)
b_A_half = tl.where(mask_half, b_A_half * gate_half, 0.0)

tl.store(p_temp_A_half, b_A_half)
tl.debug_barrier()
b_A_full = tl.load(p_temp_A_full)
b_o_dot = tl.dot(b_A_full, b_v_full)

tl.store(p_out_half, combine(b_o_half, b_o_dot_half), boundary_check=(0, 1))
```

### Pattern D: split `+=` and `tl.trans` explicitly

```python
# Before (harder to reason about lane ownership):
b_dq += tl.dot(b_ds, b_k)

# After (explicit cube/vector phases):
b_add = tl.dot(b_ds, b_k)   # cube
b_dq = b_dq + b_add         # vector (must be split by lane ownership)
b_kt = tl.trans(b_k)        # vector
b_qk = tl.dot(b_q, b_kt)    # cube
b_ds2 = b_ds * b_qk         # vector
```

## Common mistakes

### Mistake A: conflicting autotune meta-parameters

Wrong: `k[grid](..., BT=128, H=H)` when `BT` is already in `triton.Config`.

Right: pass `BT` only from `Config`; call site omits it.

### Mistake B: lane-0 fallback

Wrong: real computation only in `if vec_id == 0:`, lane 1 does nothing.

Right: both lanes own disjoint half via `half_off`, both call `tl.store`.

### Mistake C: missing barrier before full-tile cube load

Wrong: `tl.store(p_tmp_half, b_half)` immediately followed by `tl.load(p_tmp_full)`.

Right: insert `tl.debug_barrier()` between store and load.

### Mistake D: assuming partial dot equals full-dot slice

Wrong: `b_a_half = tl.dot(b_k_half, tl.trans(b_k_full))`.

Right: `b_a_full = tl.dot(b_k_full, tl.trans(b_k_full))`, then split rows with fixpipe or staged load.

## Pre-merge checklist

- [ ] lane ownership is disjoint and symmetric
- [ ] no lane-0-only computational fallback
- [ ] full-tile `tl.dot` semantics preserved where required
- [ ] vanilla `make_block_ptr` regions remain `make_block_ptr` in `_sub_vec`
- [ ] all split stores use disjoint half-tile ownership (no overlap races)
- [ ] hidden vector ops (`+=`, `tl.trans`) are decomposed where needed
- [ ] handoff synchronization is explicit and correct
- [ ] scratch allocation is sufficient for runtime/tile sweep
- [ ] accuracy passes against golden
- [ ] perf measured and recorded
- [ ] vanilla and `_sub_vec` paths remain clearly separable
