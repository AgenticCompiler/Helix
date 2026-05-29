# Report Hardware Prompt Design

## Summary

- Stop treating `env-info.json` as a required optimize artifact for report generation.
- Inject hardware environment information into the report agent prompt instead.
- Keep `target_chip` explicit for optimize, while report commands derive hardware info directly from local probing.
- Keep slow report-agent startup out of unrelated optimize tests by disabling auto-report in tests unless the report path is under test.

## Problem

The original report flow expected hardware metadata to come from a persisted
workspace file. That created two issues:

- it added an extra optimize artifact that users may not want to keep
- tests that accidentally launched the report agent became slow and flaky

At the same time, direct `report` and `report-batch` invocations must still have
the same hardware context that automatic post-optimize reporting sees, without
requiring a separate chip flag.

## Goals

- Make report generation rely on prompt-provided hardware info instead of a
  persisted `env-info.json` artifact.
- Keep direct `report`, `report-batch`, and post-optimize auto-report behavior aligned.
- Add focused tests around the report path so optimize tests do not need to
  cover report internals indirectly.

## Non-Goals

- Do not redesign the report skill output format.
- Do not change optimize success semantics when report generation fails.
- Do not add a new persistent hardware metadata artifact.

## Design

### Prompt-Sourced Hardware Info

`report` prompt construction should append a `Hardware environment information`
section that includes:

- `chip_name` when available
- `cann_version` when available
- `driver_version` when available

### CLI Surface

`report` and `report-batch` should not accept `--target-chip`. They should rely
on local hardware probing for the report prompt's environment section.

### Testing

- Add report-focused regression tests for prompt injection and the absence of a
  report-side `--target-chip` flag.
- Keep existing optimize tests fast by passing `--no-report` unless a test is
  specifically validating post-optimize reporting behavior.
