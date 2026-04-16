## Summary

- Rename the optimize repair reference from `contracts.md` to `round-failure-handling.md`.

## User-Visible Behavior

- The optimize skill should describe this reference as guidance for round failure handling, not as a generic contract.
- The reference should clearly cover failed correctness checks, failed benchmark execution, and slower-but-correct rounds within one optimize round lifecycle.

## Implementation Notes

- Rename `skills/triton-npu-optimize/references/contracts.md` to `skills/triton-npu-optimize/references/round-failure-handling.md`.
- Update optimize skill and workflow links to the new filename.
- Update the reference title and any design docs that enumerate optimize reference files so the naming stays internally consistent.
