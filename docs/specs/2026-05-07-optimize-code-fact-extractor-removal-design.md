# Optimize Code Fact Extractor Removal Design

## Summary

Remove `skills/triton/triton-npu-optimize/scripts/extract_code_facts.py` and the prompt guidance that treats it as a pattern-triage helper.

The current helper only emits two narrow facts, both based on ad hoc AST heuristics. That is not broad enough to justify a permanent routing utility, and it is not valuable enough to keep as a first-class optimize aid.

## Goal

Keep pattern triage lightweight and evidence-driven without maintaining a weak code-fact extractor.

## Scope

- Delete the helper script.
- Remove its mention from optimize prompts and skill guidance.
- Update tests to reflect that pattern triage now relies on direct code inspection plus the generated pattern index.

## Validation

- Update prompt and contract tests so they no longer expect the extractor.
- Ensure optimize guidance still tells the agent to inspect code structure directly when pattern triage is unclear.
