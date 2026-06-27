---
id: al-sync
---

# al.sync_block_set/wait Pipeline Protocol

## Summary

Add explicit producer-consumer synchronization between cube and vector `al.scope` blocks using `al.sync_block_set` and `al.sync_block_wait`, covering single-buffer, ping-pong, and triple-buffer (task-ring) pipeline depths.

## Use When

- A kernel uses separate `al.scope(core_mode="cube")` and `al.scope(core_mode="vector")` blocks that share buffer data across the boundary.
- Data races, hangs, or incorrect results appear between cube and vector scope handoff points.
- The kernel needs throughput improvement by adding ping-pong buffering (`PIPE_STAGES=2`).
- The kernel uses preload or task-ring patterns (`PIPE_STAGES >= 3`) requiring doubled sync events.
- L0C result buffers are bound before `al.fixpipe` transfer and need synchronization with vector UB consumption.

## Avoid When

- The kernel does not cross-scope buffer handoff (single scope, no cube/vector split).
- Synchronization is entirely handled by the compiler with `disable_auto_inject_block_sync=False`.

## Signals

### Code

- `al.scope(core_mode="cube")` and `al.scope(core_mode="vector")` blocks exist but no `al.sync_block_set` or `al.sync_block_wait` calls are present.
- `bl.alloc` buffers are written in one scope and read in the other without explicit sync.
- `al.fixpipe` or `al.copy` is used without corresponding sync set/wait pairs.

### Profile

- Kernel exhibits intermittent wrong results or hangs that vary with tile size.
- Performance is lower than expected for multi-buffer configurations, suggesting sync stalls.

## Related Patterns

- [al-scope](al_scope.md) — sync events are always needed between cube and vector scopes.
- [al-copy-fractal](al_copy_fractal.md) — `al.fixpipe` and `al.copy` are the data movements that sync protects.
- [sub-vec-id-1to2](sub_vec_id_1to2.md) — sub_vec_id lane split uses L1 subview handoff that requires sync.

## What To Verify After Applying

- Every `al.sync_block_set` has a matching `al.sync_block_wait` with identical `(producer, consumer, event_id, src_pipe, dst_pipe)`.
- Producer/consumer strings are not swapped: set is called by the producer, wait is called by the consumer.
- Pre-loop credit initialization (`cube_prefree_p_l1`, `vec_prefree_s_ub`, `vec_prefree_pv_ub`) is present for ping-pong protocols.
- Post-loop drain is present for ping-pong protocols.
- Event IDs are unique per (producer→consumer, buffer resource) pair within the same pipeline stage.
- `is_mem_unique=True` is set on L0C allocations and on UB buffers that must not alias.
- For PIPE_STAGES >= 3, each sync call is issued twice.

---

# Ascend affinity API: sync reference

## API

```python
al.sync_block_set(producer, consumer, event_id, src_pipe, dst_pipe)
al.sync_block_wait(producer, consumer, event_id, src_pipe, dst_pipe)
```

- `producer` / `consumer`: string, `"cube"` or `"vector"`
- `event_id`: integer, unique per pair per pipeline stage
- `src_pipe`: pipe that has completed on the producer side
- `dst_pipe`: pipe the consumer is waiting to start

**`set`** is called by the producer after a transfer completes.  
**`wait`** is called by the consumer before it touches the transferred data.

Every `set` must be matched by exactly one `wait` with the same `(producer, consumer, event_id, src_pipe, dst_pipe)` signature.

## Pipe channels

| Channel | Meaning |
|---|---|
| `al.PIPE.PIPE_FIX` | Fixpipe (cube→UB DMA) |
| `al.PIPE.PIPE_V` | Vector operation |
| `al.PIPE.PIPE_MTE1` | MTE1 — L1/global DMA on the cube side |
| `al.PIPE.PIPE_MTE3` | MTE3 — L1/global DMA on the vector side |

## Memory locations

| Symbol | Full name | Who reads/writes |
|---|---|---|
| L0C | Level-0 C register file | Cube (output of `tl.dot`) |
| UB | Unified Buffer (on-chip SRAM) | Both cube (via fixpipe) and vector |
| L1 | Level-1 cache | Shared staging between cube and vector |
| Global (HBM) | Off-chip DRAM | `tl.load` / `tl.store` |

`bl.alloc` address spaces:
- `al.ascend_address_space.UB` — UB allocation
- `al.ascend_address_space.L1` — L1 allocation
- `al.ascend_address_space.L0C` — L0C allocation (cube output, requires `is_mem_unique=True`)

## Single-buffer protocol (fa_fwd_Affinity)

Three events form the QK→softmax→PV chain:

```
cube:  QK done → set(cube, vector, 0, PIPE_FIX, PIPE_V)
vector:          wait(cube, vector, 0, PIPE_FIX, PIPE_V) → softmax → copy P→L1
vector:  P ready → set(vector, cube, 1, PIPE_MTE3, PIPE_MTE1)
cube:             wait(vector, cube, 1, PIPE_MTE3, PIPE_MTE1) → PV done
cube:  PV done → set(cube, vector, 2, PIPE_FIX, PIPE_V)
vector:          wait(cube, vector, 2, PIPE_FIX, PIPE_V) → acc update
```

Full code from `_attn_fwd`:
```python
# cube scope
_qk_matmul(q, K_block_ptr, qk_ub, ...)
al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
_pv_matmul(p_l1, pv_ub, V_block_ptr, ...)
al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

# vector scope
al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
... softmax, copy p→L1 ...
al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
... acc update ...
```

## Ping-pong protocol (fa_fwd_parallel)

Twelve events handle ping-pong buffering (PIPE_STAGES=2). Event IDs are constants; the ping/pong selection is done with `(sid & 1)` or `(pvid & 1)` on the buffer pointer.

Defined as named helper functions (keep them as free-standing `@triton.jit` functions for clarity):

```python
# Vector signals "s_ub is free" to cube before cube writes QK
@triton.jit
def vec_prefree_s_ub():
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# Vector signals "pv_ub is free" to cube before cube writes PV
@triton.jit
def vec_prefree_pv_ub():
    al.sync_block_set("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# Vector signals "p→L1 copy done" to cube
@triton.jit
def vec_postwait_p_l1():
    al.sync_block_wait("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

# Cube signals "p_l1 buffer is free for vector to write"
@triton.jit
def cube_prefree_p_l1():
    al.sync_block_set("cube", "vector", 6, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)

# Cube waits for "s_ub is ready from vector"
@triton.jit
def cube_postwait_s_ub():
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# Cube waits for "pv_ub is ready from vector"
@triton.jit
def cube_postwait_pv_ub():
    al.sync_block_wait("vector", "cube", 10, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

Inside the loop, the per-iteration events are:

| ID | Meaning | Caller |
|---|---|---|
| 0 | QK+fixpipe done (qk_ub ready) | cube sets, vector waits |
| 2 | s_ub free (vector done with qk_ub) | vector sets, cube waits |
| 4 | p→L1 copy done | vector sets, cube waits |
| 6 | p_l1 consumed (P·V done) | cube sets, vector waits |
| 8 | PV fixpipe done (pv_ub ready) | cube sets, vector waits |
| 10 | pv_ub free (vector done with pv_ub) | vector sets, cube waits |

Pre-loop initialization pattern:
```python
# in cube scope, before the loop:
cube_prefree_p_l1()      # signal p_l1 is initially free

# in vector scope, before the loop:
vec_prefree_s_ub()       # signal s_ub is initially free
vec_prefree_pv_ub()      # signal pv_ub is initially free
```

Post-loop drain pattern:
```python
# in cube scope, after the loop:
cube_postwait_s_ub()
cube_postwait_pv_ub()

# in vector scope, after the loop:
vec_postwait_p_l1()
```

## Triple-buffer protocol (fa_fwd_preload)

For PIPE_STAGES=3 / task-ring preload, each sync call is issued **twice** to cover the extra pipeline depth:

```python
@triton.jit
def vec_prefree_s_ub():
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)  # twice
```

This pattern applies to all six helper sync functions. The event IDs (0, 2, 4, 6, 8, 10) remain the same.

## L0C binding and fixpipe chain

When using L0C staging for the cube result before fixpipe:

```python
# Inside cube scope / _qk_matmul:
qk = tl.dot(q, trans_k)
bl.to_buffer(qk, bind_buffer=qk_l0c)          # bind result to L0C alias
al.sync_block_wait("vector", "cube", 2, ...)   # wait for UB free
qk_ub_tensor = bl.to_tensor(qk_ub)
al.fixpipe(qk, bl.to_buffer(qk_ub_tensor, al.ascend_address_space.UB),
           al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)
al.sync_block_set("cube", "vector", 0, ...)    # signal UB ready
```

L0C alloc must use `is_mem_unique=True`:
```python
qk_l0c = bl.alloc(tl.float32, [BLOCK_M, BLOCK_N], al.ascend_address_space.L0C, is_mem_unique=True)
```

`is_mem_unique=True` is also required on `pv_ub` buffers in the preload kernel for correct alias analysis.

## Common mistakes

**Wrong**: calling `set` and `wait` with swapped producer/consumer strings.
- `al.sync_block_set("cube", "vector", ...)` is called by cube; `al.sync_block_wait("cube", "vector", ...)` is called by vector.
- The strings in `wait` must match the strings in `set` exactly.

**Wrong**: omitting the pre-loop free signals (`cube_prefree_p_l1`, `vec_prefree_s_ub`, `vec_prefree_pv_ub`).
- These initialize the pipeline's credit count. Without them, the first iteration deadlocks.

**Wrong**: using the same event ID for two different buffer transfers in the same pipeline stage.
- Each (producer→consumer, buffer resource) pair needs its own event ID.
