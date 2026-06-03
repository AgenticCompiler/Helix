# Skill-First Agent Prompt Design

## Problem

Some agent-facing prompts mention helper subcommands such as `check-round` directly without first naming the owning skill. That wording is fine inside the owning skill when the command is self-contained, but it is brittle in top-level prompts or cross-skill handoffs because the agent may not know which workflow contract the command belongs to.

## Goals

- Make top-level and cross-skill agent prompts name the owning skill before naming helper subcommands.
- Keep helper subcommands available as concrete execution details once the skill context is established.
- Preserve current CLI command names and existing intra-skill command examples.

## Non-Goals

- Do not rename CLI subcommands such as `gen-test`, `run-test`, or `check-round`.
- Do not rewrite self-contained command examples that already live inside the owning skill.
- Do not change human-facing README command documentation in this pass.

## Design

1. In generated agent prompts, treat the skill as the primary workflow contract and describe helper subcommands as actions taken through that skill.
2. In optimize prompts specifically, replace bare `check-baseline` / `check-round` instructions with wording that explicitly routes those checks through `triton-npu-optimize-submit-baseline / triton-npu-optimize-submit-round`.
3. Add prompt tests that lock in this distinction:
   - cross-skill prompt text must be skill-first
   - self-contained same-skill command references remain acceptable
