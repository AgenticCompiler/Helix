# Model Entrypoint Precedence Design

## Summary

When generation workflows inspect an operator file that contains a `torch.nn.Module` entrypoint such as `class Model`, an intermediate Triton wrapper function, and one or more kernels, the workflow should prefer the module class as the public entrypoint when the call chain is:

`Model.forward(...) -> wrapper(...) -> kernel(...)`

This keeps generated harnesses aligned with the user-facing API instead of anchoring them to an internal helper wrapper.

## Problem

The current generation contract says generated harnesses should target the resolved public entrypoint, but it does not explicitly disambiguate the common pattern where:

- a `class Model` is the real PyTorch-facing API
- `Model.forward()` delegates to a wrapper function
- the wrapper function launches the Triton kernel

Without an explicit rule, an agent can incorrectly choose the wrapper function as the harness API, which skips the intended module boundary and can misrepresent the operator contract.

## Decision

Add one explicit precedence rule for generation workflows:

- when a `class Model` or equivalent no-argument `torch.nn.Module` clearly wraps an intermediate Triton wrapper function that then launches the kernel, prefer the module class as the resolved public entrypoint
- do not select the intermediate wrapper function in that pattern unless the module is not a safe or valid public entrypoint under the existing contract

This is a narrow refinement, not a global inversion of all entrypoint priorities.

## Scope

Update the generation-side contract only:

- `skills/triton-npu-gen-test/SKILL.md`
- `skills/triton-npu-gen-bench/SKILL.md`
- generation prompts sent by the CLI
- focused generation-entrypoint design notes that currently describe the older precedence

Do not change CLI flags or runtime harness loading behavior in this work.

## Verification

- add a generation contract test that requires the skill docs to state the `Model -> wrapper -> kernel` preference explicitly
- add a generation prompt test that requires the `gen-test` prompt to carry the same instruction
