# Atomic Contention Owner-Computes Store

## Summary

Replace many-program atomic updates to a small output domain with an owner-computes decomposition: transpose the grid from input tiles to output targets, let each program own one bucket or reduction target, scan the contributing input region, and write the final value with a plain `tl.store`.

## Use When

- The hot kernel updates a small or moderate output domain with `tl.atomic_add`, `tl.atomic_max`, or a similar atomic operation from many programs.
- Multiple programs can write the same output address, such as histogram bins, class buckets, segment IDs, sparse row buckets, or other low-cardinality reduction targets.
- Profiling or benchmark scaling indicates atomic/store-side contention is more expensive than rereading the input for each owner program.
- The output target can be partitioned so exactly one Triton program owns each bucket, output row, segment, or reduction slot.
- The per-owner scan can be expressed with regular `tl.load`, vector predicates, and `tl.sum` / `tl.max` / another associative reduction.
- The output cardinality is small enough that the extra read traffic, roughly `output_targets * input_extent`, is plausible for the benchmark shape range.
- The aggregation is order-independent under the operator's reference tolerance, such as count, integer sum, max without order-dependent tie-breaking, or floating-point sum where reordered accumulation is acceptable.

## Avoid When

- The output domain is large enough that owner-computes would multiply global reads beyond the cost of atomics.
- Atomic collisions are rare because output addresses are already well distributed.
- The reduction target cannot be uniquely owned without changing semantics or adding a second merge phase.
- The reduction operation is not associative enough for a reordered scan, or the reference requires a specific update order.
- The current bottleneck is ordinary contiguous load bandwidth, dtype conversion, scalar address generation, or wrapper overhead rather than atomic/store contention.
- Simulator output is the sole signal AND it does not show the asymmetric atomic MTE3 profile (low instr% + high cycles%). Single-program simulation cannot quantify the multi-program contention severity or predict the owner-computes payoff by itself; combine simulator evidence with source structure and benchmark scaling.
- The bucket assignment is expensive to recompute for every owner program, for example when it depends on complex running state rather than a pure function of an input element and the owner ID.

## Signals

### Code

- **Semantic essence — scatter-reduce to a small target set.** The kernel implements a two-step operation: (1) **dispatch** — each input element is routed to one output slot by a pure function of its value (bucket index, segment ID, class label, row key, etc.); (2) **reduce** — each output slot aggregates all elements routed to it with an **associative** operator (count, sum, max, min, logical-or, etc.). The defining structural property is **`output_cardinality << input_cardinality`**: many input elements compete to write the same output address, and that write contention is what `tl.atomic_*` exists to serialize and what owner-computes eliminates.
- Operators with this semantics include (non-exhaustive; only `histogram`/`bincount` is backed by the worked example below, the rest are inferred from matching semantics):

  | Operator | Dispatch key | Reduce op | Typical output cardinality |
  |----------|--------------|-----------|----------------------------|
  | `histogram` / `bincount` (worked example) | value range bucket | count | tens–hundreds |
  | `segment_sum` / `segment_max` (inferred) | segment ID | sum / max | number of segments |
  | `scatter_add` / `index_add` (inferred) | index tensor | sum | output dim size |
  | class voting / label counting (inferred) | predicted class | count | number of classes |
  | sparse row reduction (inferred) | row index | sum / max | number of rows |

  If the kernel matches this semantic shape — regardless of variable names, grid literal, or whether it literally calls `tl.atomic_*` — it is a candidate.
- **Concrete code patterns (detection hints):**
  - Grid is launched over the **input** dimension, e.g. `grid = (triton.cdiv(n_elements, BLOCK_SIZE),)`, while writes target the **output/slot** dimension.
  - Each program computes a slot index per element, then `tl.atomic_add(out_ptr + slot, ...)` / `tl.atomic_max(...)` — the atomic write is the only store to the output.
  - The wrapper zero-initializes the output buffer before launch (because the kernel incrementally accumulates via atomics); after this rewrite the zero-fill is no longer needed for accumulation.
  - The slot-membership predicate for one output slot can be evaluated by scanning input tiles, e.g. `in_slot = valid & (key == slot_id)` or a range test `value ∈ [bin_left, bin_right)`.
- **Grid-vs-output cardinality (applicability test).** Even when the semantics match, owner-computes only pays off when write contention is high. Compare **total atomic write attempts** against **distinct output addresses**:
  - **Total atomic write attempts ≫ output cardinality** (e.g. 4096 input elements each producing one `tl.atomic_add` into 256 bins → ~16 writers per bin on average) → many writers collide → **strong fit**.
  - Total atomic write attempts comparable to or smaller than output cardinality (e.g. scatter-add of 1024 values into a million-element output → ~0.001 writers per slot) → collisions are sparse → **weak fit**, prefer keeping the atomic path or private-bin decomposition.
  - "Atomic write attempts" means the number of masked-in lanes across all programs that reach the `tl.atomic_*` call — roughly `sum over programs of (valid element count per program)`. It is the collision pressure that matters, not the raw `grid_size` alone.

### Profile

All signals below are **corroborating**, not gating: the source-level semantic match (above) is the primary trigger. Use these to confirm the atomic RMW cost is actually the dominant bottleneck.

- **Simulator `report.txt` — asymmetric atomic MTE3 profile.** Single-program simulation **cannot reproduce the multi-program serialization** part of atomic contention, but it **can and does reveal the atomic operation's own MTE3 cost**, which is usually the dominant component when the output domain is small. Distinguish two layers:

  | Layer | What it is | Visible in single-program simulation? |
  |-------|------------|----------------------------------------|
  | Atomic RMW cost per se | Each `tl.atomic_*` forces a load-modify-store round-trip on MTE3 against a narrow target address set. | **Yes** — appears as a strong MTE3 signal. |
  | Cross-program contention | Multiple programs serializing on the same output addresses. | **No** — single-program traces have no competing writers. |

  When the source contains `tl.atomic_*` targeting a small output domain, the simulator typically shows:
  - **Low `MTE3 instr%`** (often single-digit, e.g. 1-2%) but **very high `MTE3 cycles%`** (e.g. > 40%, commonly > 70%). A few instructions consume most of the cycles because each atomic RMW is expensive.
  - `[Source Code Info]` attributes a large cycle share to the `tl.atomic_*` source line or to an `internal` block around it (the atomic implementation), with `MTE3` dominating that block's `[Pipe Distribution Over Cycles]`.
  - Useful contrast: in the available simulator samples, atomic-to-small-domain kernels show MTE3 cycles% in the 70-80% range, while non-atomic store-heavy kernels of similar data-movement shape stay around 5-14%. The large gap makes the asymmetric MTE3 profile a strong discriminator — but treat it as corroborating evidence tied to the source-level atomic check, not as a standalone threshold.

  This MTE3 asymmetry is a reliable **corroborating** signal: once the source semantics match, seeing it strongly confirms the atomic RMW cost is the dominant bottleneck. The "Avoid When: simulator-only evidence is the sole signal" caveat refers to the *contention severity* and the *owner-computes payoff* — you cannot read from the simulator alone how much the multi-program serialization adds on top.
- **Interaction with `scalar-vector-simulation-signal`** — when the source matches the scatter-reduce semantics, treat this pattern as the primary direction and **do not route into scalar-vector-simulation-signal Cat 1-5 first**. Those categories commonly co-fire here as *consequences* of the atomic bottleneck:
  - **Cat 4 (low VECTOR utilization)** fires because atomic MTE3 stalls star the VECTOR unit — the low utilization is a symptom, not the root cause.
  - **Cat 1 (high SCALAR instr%)** may co-fire from bucket-index address computation, but removing scalar overhead alone will not fix the atomic MTE3 dominance.
- **Hardware profiling** attributes the hot kernel to store/atomic-related pressure, such as dominant MTE3 cycles or a source line around `tl.atomic_*`.
- **Benchmark scaling**: increasing `BLOCK_SIZE` or otherwise reducing program count improves performance but leaves the same store-side bottleneck; latency grows poorly with input size when many programs update the same few output addresses.
- **Post-rewrite confirmation**: a trial owner-computes version reduces atomic instructions to zero, MTE3 cycles drop sharply (e.g. 79% → 2%), and the bottleneck shifts toward vector comparison/reduction or regular loads.

## Decision Chain

1. **Source-level semantic match (primary trigger)**: does the kernel implement scatter-reduce (dispatch + associative reduce) into a small output domain? If yes, this pattern is a primary candidate — no simulator evidence required to start.
2. **Simulator corroboration (strengthens confidence)**: does `report.txt` show the asymmetric MTE3 profile (few MTE3 instructions, dominant MTE3 cycles)? Confirms the atomic RMW cost is actually the bottleneck rather than incidental.
3. **Benchmark scaling (confirms contention, not bandwidth)**: does reducing program count help but leave the store-side bottleneck? Rules out pure bandwidth-bound cases.
4. **Post-rewrite simulator check (secondary)**: after applying owner-computes, the simulator is useful to verify MTE3 dropped sharply and to find the new bottleneck (typically VECTOR comparison/reduction, or SCALAR from the per-owner scan loop).

## Optimization Strategy

1. Identify the output ownership key.

   Choose the output dimension that causes atomic collisions: bucket, segment ID, class ID, row, or another small reduction target.

2. Change the grid to output ownership.

   Launch one program, or one small fixed set of programs, per output target. For a bucketed-count kernel, this usually means `pid = tl.program_id(0)` is the bucket ID and `grid = (num_buckets,)`.

3. Scan the contributing input range inside each owner.

   Iterate over `n_elements` or the relevant input extent in `BLOCK_SIZE` tiles. Use vector predicates to keep only values that contribute to the owner target.

4. Accumulate locally.

   Use a scalar or vector reduction accumulator inside the program. For counts, accumulate `tl.sum(mask.to(tl.int32))`. For sums, accumulate `tl.sum(tl.where(mask, values, 0))`.

5. Store once.

   Because ownership guarantees one writer per target, replace `tl.atomic_*` with `tl.store(out + owner_id, final_value)`.

6. Retune tile size after the structural rewrite.

   The optimal `BLOCK_SIZE` changes after atomics are removed. Larger tiles may reduce loop overhead, but tiny shapes may regress from over-provisioned lanes.

## Tradeoff Model

The rewrite exchanges fewer synchronization points for more regular reads:

| Structure | Reads | Atomic operations | Normal stores |
|-----------|-------|-------------------|---------------|
| Input-owner atomic | `N` | up to `N` | 0 or few |
| Output-owner scan | `num_targets * N` | 0 | `num_targets` |

The owner-computes form wins when repeated regular reads are cheaper than serialized atomic updates to a small output domain. Rough magnitude guidance (the `num_targets` tens–hundreds band is backed by the worked example below; the thousands-band regression and the bandwidth-bound case are reasoned from the read-traffic model, not yet measured — validate per shape):

- `num_targets` in the tens to low hundreds and `input_extent` up to a few thousand → owner-computes won in the worked example (histogram, bins 50-256), and the read-traffic model suggests this band is generally favorable.
- `num_targets` in the thousands → the read-traffic model predicts owner-computes starts losing because `num_targets * input_extent` dominates; prefer private-bin/private-row decomposition (each program owns a disjoint output row, reduce after) or keep the atomic path. Treat this as a hypothesis to benchmark, not a proven boundary.
- `input_extent` very large and already pure bandwidth-bound → even a small `num_targets` may not amortize; check benchmark scaling first.

When shape ranges are broad, keep this as a shape-dispatched path rather than replacing the atomic path unconditionally. For example, use owner-computes for low-cardinality bucket counts and keep the atomic or hierarchical path for high-cardinality buckets.

### Related strategy: private-bin / private-row decomposition

Owner-computes is the most aggressive atomic-elimination rewrite (zero atomics, zero contention, one writer per target). A weaker but sometimes cheaper alternative is **private-bin decomposition**: each program writes to its own disjoint output row (`out + pid * stride + bucket`) using atomics only against a private region, then a second Triton kernel uses `tl.sum` across the per-program rows to produce the final output. This eliminates *cross-program* contention while keeping per-program atomics. It is preferable when the output domain is too large for owner-computes to amortize the reread, or when the input scan per owner would exceed UB capacity.

## Example: Histogram

### Problem (baseline: input-tile grid + atomic)

The baseline maps programs to input tiles. Each program computes a target bucket for each loaded element, then atomically increments the shared bucket output. When many elements map to the same few buckets, many programs serialize on the same output addresses.

```python
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
```

```python
pid = tl.program_id(0)
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offs < n_elements
keys = tl.load(key_ptr + offs, mask=mask, other=-1)
valid = mask & (keys >= 0) & (keys < num_buckets)
tl.atomic_add(counts_ptr + keys, 1, mask=valid)
```

Observed on a histogram benchmark (5 cases, `n_elements` 128-4096, `bins` 50-256, BLOCK_SIZE=256 → grid 1-16 programs): the larger cases (grid 4-16 programs writing masked atomic lanes into 64-256 bins) are where contention bites. Baseline simulator `report.txt` showed the asymmetric atomic MTE3 profile: `MTE3 instr% = 1.8%` but `MTE3 cycles% = 79.1%`; the `tl.atomic_add` source line attributed ~33% of cycles, the `internal` atomic block ~65% (MTE3 97% within it). `scalar-vector-simulation-signal` Cat 4 also fired (VECTOR util 0.7%) as a downstream symptom.

### Rewrite (owner-computes: output-bucket grid + plain store)

Transpose the grid to output ownership. Each program owns one bucket, scans input tiles, counts only elements for that bucket, and stores the final count once. No other program writes the same output address.

```python
grid = (num_buckets,)
```

```python
bucket_id = tl.program_id(0)
count = 0

for start in range(0, n_elements, BLOCK_SIZE):
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    keys = tl.load(key_ptr + offs, mask=mask, other=-1)
    in_bucket = mask & (keys == bucket_id)
    count += tl.sum(in_bucket.to(tl.int32))

tl.store(counts_ptr + bucket_id, count)
```

Use ordinary `range()` for scans whose trip count depends on runtime sizes. Do not convert long runtime scans to `tl.static_range()`, because full unrolling can explode IR size and compile time.

Result: MTE3 dropped to ~2%, geomean **14.55x** total-op speedup. Subsequent `BLOCK_SIZE` tuning (256 → 4096) pushed it further by cutting the per-owner scan loop overhead.

Contrast: a private-histogram decomposition (per-program output row + a second `tl.sum` kernel) on the same operator achieved only ~1.9x because it kept per-program atomics and added a reduction pass — illustrating that owner-computes wins when `num_targets` is small enough to make the reread cheap.

## Boundary And Semantics Notes

- For range-bucketed operators, reproduce reference interval semantics exactly. A common shape is half-open buckets `[left, right)` with a closed final bucket that includes the maximum value.
- Preserve filtering rules for invalid values, NaN, out-of-range keys, masked tail lanes, and empty inputs.
- If the original atomic path accumulates floating-point values, confirm that reordered accumulation is acceptable for the operator's tolerance.
- If the output must be returned in the input dtype, separate the algorithmic rewrite from dtype-cast cleanup. First remove atomics, then optimize casts with a constexpr flag or dispatch if profiling shows conversion overhead.

## What To Verify After Applying

- Correctness for boundary buckets, invalid keys, NaN/filtering rules, empty inputs, dtype conversion, and exact reference edge cases.
- Representative shapes where `output_targets * input_extent` is both small and large; the owner-computes path may need shape dispatch.
- Perf confirms atomic operations disappeared and total-op latency improves against the immediate parent.
- Profiling confirms the new bottleneck is not worse regular-load bandwidth, scalar conversion, or a new dtype-cast path.
- Follow-up tuning separately evaluates `BLOCK_SIZE`, `num_warps`, software pipelining, and dtype-specialized casts. Do not mix these follow-up gains into the evidence for the structural rewrite.

## Related Patterns

- `scalar-vector-simulation-signal` — its Cat 1/4 commonly co-fire as downstream symptoms of atomic MTE3 pressure; this pattern takes priority when source contains `tl.atomic_*`.
- `software-pipeline-dependency-profiling`
- `autotune`
- `scalar-latency-traps`
- `effective-extent-tiling`
- `program-multiple-rows`
