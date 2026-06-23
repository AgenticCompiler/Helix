# TileLang Compute & Data Movement

## Data movement

```python
T.copy(src, dst)   # GM→UB, GM→L1, UB→GM, L1→L0A, L1→L0B, L0C→GM
T.barrier_all()    # Synchronization barrier between copy and compute
```

Supported copy directions: GM↔UB, GM↔L1, UB↔L1, UB→L1, UB→UB, L1→L0A, L1→L0B, L0C→GM.

```python
T.tile.atomic_add(dst_gm, src_local)  # Atomic accumulate local → GM
```

## Element-wise (developer mode)

```python
for i, j in T.Parallel(M, N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]    # +, -, *, /
    c_ub[i, j] = T.exp(a_ub[i, j])           # T.exp, T.max, T.min, T.sqrt
```

## Element-wise (expert mode)

| API | Operation |
|-----|----------|
| `T.tile.add(dst, src0, src1)` | `dst = src0 + src1` |
| `T.tile.sub(dst, src0, src1)` | `dst = src0 - src1` |
| `T.tile.mul(dst, src0, src1)` | `dst = src0 * src1` |
| `T.tile.div(dst, src0, src1)` | `dst = src0 / src1` |
| `T.tile.max(dst, src0, src1)` | `dst = max(src0, src1)` |
| `T.tile.min(dst, src0, src1)` | `dst = min(src0, src1)` |
| `T.tile.exp(dst, src)` | `dst = exp(src)` |
| `T.tile.sqrt(dst, src)` | `dst = sqrt(src)` |
| `T.tile.rsqrt(dst, src)` | `dst = 1/sqrt(src)` |
| `T.tile.relu(dst, src)` | `dst = max(0, src)` |
| `T.tile.leaky_relu(dst, src, scalar)` | Leaky ReLU |
| `T.tile.sin(dst, src)` / `T.tile.cos(dst, src)` | Sine / Cosine |
| `T.tile.abs(dst, src)` | Absolute value |
| `T.tile.ln(dst, src)` | Natural log |

## Matrix multiply (Cube)

```python
T.gemm_v0(A_L1, B_L1, C_L0, transpose_A=False, transpose_B=False, init=(k==0))
```

- `A_L1`, `B_L1`: L1 buffers
- `C_L0`: L0C accumulator fragment
- `init=True` on first k-iteration to zero accumulator

## Reduction

```python
T.reduce_sum(src, dst, dim=-1, clear=True)
T.reduce_max(src, dst, dim=-1, clear=True)
T.reduce_min(src, dst, dim=-1, clear=True)
```

`clear=False` accumulates into existing `dst` values.

Variant gemm form: `T.gemm_v` (no `_0` suffix) accepts the same parameters.

## Control flow

```python
result = T.if_then_else(cond, true_value, false_value)  # TIR conditional expression
```

## Type reinterpretation

```python
T.reinterpretcast(dst, src, "float")  # Reinterpret src bits as float in dst
```

## Other operations

| API | Purpose |
|-----|---------|
| `T.tile.cast(dst, src, mode, count)` | Precision cast (modes: `CAST_RINT`, `CAST_FLOOR`, `CAST_CEIL`, etc.) |
| `T.tile.transpose(dst, src)` | 16×16 block transpose |
| `T.tile.fill(buffer, value)` | Fill buffer with scalar |
| `T.tile.compare(dst, src0, src1, mode)` | Compare (modes: `EQ`, `NE`, `GT`, `GE`, `LT`, `LE`) |
| `T.tile.select(dst, mask, src0, src1, mode)` | Masked element selection |
| `T.tile.gather_mask(dst, src, pattern)` | Patterned element gather |
| `T.tile.axpy(dst, src, scalar)` | `dst = scalar * src + dst` |
| `T.tile.bitwise_and/or/not/xor` | Bitwise ops |
| `T.tile.bitwise_lshift/rshift` | Bit shift ops |
| `T.tile.createvecindex(dst, first)` | Create vector index starting at `first` |
