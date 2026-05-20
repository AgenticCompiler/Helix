# unsupported-dtype-fallback

## Summary

The hot path appears to be running an operator on AiCPU or another slower fallback backend because the active dtype or operator combination is unsupported on AiCore.

## Evidence To Confirm

- Runtime logs explicitly mention unsupported dtype or unsupported operator placement on **AiCore**, often together with an **AiCpu** fallback warning.
- Profiling shows an operator with tiny logical work still taking unusually high time, which suggests fallback dispatch and synchronization overhead dominate the measurement.
- Code inspection shows a framework op using **`int32`**, **`int64`**, or another non-default dtype where a semantically equivalent supported dtype may exist for the active workload.

## Candidate Pattern Directions

- `argsort-avoid-aicpu-fallback`

## Common Non-Matches

- Not every AiCPU operator in a profile is a meaningful optimization target; some tiny host-side setup ops are expected and not on the critical path.
- A slow operator without an explicit fallback warning is not enough evidence by itself; confirm backend placement before assuming a dtype-support gap.
