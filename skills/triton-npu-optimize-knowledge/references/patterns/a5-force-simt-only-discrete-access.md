# A5 SIMT-Only Discrete Access Pattern

## Summary

On A5, when profiler evidence shows a discrete-memory-access Triton kernel is dominated by AIV scalar work rather than vector or Cube work, try launching the kernel with `force_simt_only=True`, then retune launch parameters such as `num_warps` and grid decomposition.

This is a profile-gated launch-mode experiment, not a default setting. Apply it only after code inspection or model analysis indicates the kernel is mostly discrete/index-driven memory access rather than regular vector arithmetic or Cube-heavy matmul.

## A5 Confirmation

Treat A5 as a required precondition, not an inference from scalar-heavy profiling alone.

Accept any of these as A5 evidence:

- User explicitly states the target is A5.
- `msprof`/profile metadata, such as `PROF_*/device_*/sample.json` or `PROF_*/host/sample.json`, names an A5/Ascend 950-class target.
- Runtime or compile logs name an A5/Ascend 950-class target, for example `ascend950PR` or `ascend950DT`.
- Runtime device queries such as `torch.npu.get_device_name(0)` or `torch.npu.get_device_properties(0)` report an A5/Ascend 950-class device.
- Environment/CANN target settings clearly name an A5/Ascend 950-class SOC.

If A5 cannot be confirmed, record this pattern as a candidate only. Do not apply `force_simt_only=True` solely because `aiv_scalar_ratio` is high.

## Use When

- Target hardware is confirmed as A5 by user statement, profile metadata, runtime/compile logs, runtime device query, or environment/CANN target settings.
- `msprof` profiling has an `op_summary_*.csv` row whose `opName` matches the Triton kernel name.
- That row shows `aiv_scalar_ratio` clearly higher than `aiv_vec_ratio` and `cube_utilization`.
- The kernel body is primarily discrete/index-driven memory access, gather/scatter-like movement, or scalar-heavy pointer/index computation.
- Correctness validation and representative benchmark reruns are available after changing launch parameters.

## Avoid When

- The kernel is Cube-heavy, matmul-like, or already dominated by vector arithmetic.
- Profiling does not identify the target kernel row confidently by `opName`.
- The scalar ratio is only slightly higher or the bottleneck is host launch overhead, copy overhead, or another operator.
- The shape regime is not representative enough to justify architecture-specific launch-mode changes.

## Signals

### Profile

- In `op_summary_*.csv`, `opName` equals or clearly contains the target kernel name.
- `aiv_scalar_ratio` is much larger than `aiv_vec_ratio`.
- `aiv_scalar_ratio` is much larger than `cube_utilization`.
- Low Cube utilization is expected from the kernel semantics, not an accidental symptom of a missed `tl.dot` rewrite.

### Code

- Index arrays, indirect offsets, gather/scatter addresses, masks, or per-lane pointer reconstruction dominate the hot path.
- The computation has little dense arithmetic after each load.
- The kernel looks closer to sparse/discrete movement than SIMT-friendly dense vector math.

## Optimization Strategy

1. Confirm the profile row and kernel-name match from `op_summary_*.csv`.
2. Confirm A5 using the evidence rules above.
3. Confirm the kernel is discrete-memory-access dominated by reading the Triton kernel body.
4. Add `force_simt_only=True` to the Triton kernel launch:

   ```python
   _kernel[grid](
       ...,
       BLOCK_SIZE=BLOCK_SIZE,
       num_warps=num_warps,
       force_simt_only=True,
   )
   ```

5. Run correctness first. Do not trust performance if precision or functional comparison fails.
6. Benchmark representative shapes and compare against the parent candidate.
7. Tune launch parameters after enabling SIMT-only mode:
   - `num_warps`
   - grid decomposition
   - per-program work size
   - block size when it affects scalar/index work
8. Keep the change only if correctness passes and measured performance improves.

## What To Verify After Applying

- Correctness/precision remains within the operator's accepted tolerances for all tested dtypes and boundary shapes.
- The round record cites the A5 evidence source, such as profile metadata, log text, runtime query output, environment setting, or explicit user statement.
- `force_simt_only=True` improves representative benchmark results, not only one isolated shape.
- `num_warps` and grid choices were rechecked after changing launch mode.
- A follow-up profile confirms reduced scalar-heavy bottleneck or improved elapsed time for the target kernel row.

## Risks

- `force_simt_only=True` is architecture- and backend-sensitive; it may regress non-A5 targets or dense vector/Cube kernels.
- The best `num_warps` or grid under SIMT-only mode may differ from the previous best configuration.
- A scalar-heavy profile can point to code-structure problems that should be fixed directly instead of hidden by launch mode.
