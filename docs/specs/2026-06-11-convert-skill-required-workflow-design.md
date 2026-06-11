# Convert Skill Required Workflow Design

## Goals

- Make `skills/triton-npu-convert-pytorch-operator/SKILL.md` easier to scan during convert work.
- Separate ordered workflow steps from always-on constraints and validation rules.
- Preserve the existing conversion contract while reducing repetition across nearby sections.

## User-Visible Semantics

- `## Required Workflow` should read like an execution sequence, not a catch-all rules list.
- Permanent constraints such as source-file immutability, Ascend-only targeting, helper preservation, and no in-file test code should live outside the ordered workflow list.
- Validation guidance should make the reuse-first rule and repair loop easy to find without burying them inside the main workflow.

## Proposed Structure

- Add a short `## Core Constraints` section before `## Required Workflow`.
- Reduce `## Required Workflow` to a compact phase-oriented sequence:
  - inspect the source
  - write the converted output
  - preserve helper blocks
  - validate with the requested mode
  - finish only on success or a clear blocker
- Move reuse/generate/run/repair details into a dedicated `## Validation Flow` section.
- Trim obvious duplicates in `## Quality Rules` and `## Do Not` so those sections reinforce the contract instead of restating the same points verbatim.

## Non-Goals

- Do not relax the Triton-kernel-only forward constraints.
- Do not change the converted output example.
- Do not change the actual convert workflow contract; this is a documentation and structure cleanup only.
