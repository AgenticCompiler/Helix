---
name: npu-report
description: Generate a Chinese optimization report from an existing optimize workspace. Use when an optimize session needs a human-readable summary that synthesizes environment info, round history, performance data, and pattern analysis into a structured report.
---

# npu-report

## Purpose

Generate a human-readable Chinese optimization report (`report.md`) by reading an
existing optimize workspace. This skill is **read-only** — it reads artifacts that
were already produced by the `{language}-npu-optimize` workflow and synthesizes them
into a structured report.

## Workflow

### Step 1: Read hardware environment info

The invocation prompt includes a `Hardware environment information` section with:

- `target_chip` — target chip series (A3 or A5)
- `chip_name` — full chip model name
- `cann_version` — CANN toolkit version
- `driver_version` — Ascend driver version

If the hardware information section is absent from the prompt, note "未记录" for
the verification info section.

### Step 2: Read optimization note

Read `opt-note.md` at the root of the workspace. Parse:

- **Round history** — each `## Round N` section gives the parent, theme, result,
  best-status, and links to round-level artifacts.
- **Overall Summary** — contains `Final best round`, `Geomean speedup`,
  `Validated branches`, `Outcome`, `Next step`, and `Key optimization points`.

### Step 3: Read each round summary

For each round `N` listed in `opt-note.md`, read `opt-round-N/summary.md` for
detailed per-round optimization content:

- Optimization points with code/IR context
- Hypothesis behind the change
- Correctness and benchmark results
- Assessment of the outcome

### Step 4: Read best-round operator source and annotate

Locate the best round from `opt-note.md` Overall Summary. Read the corresponding
`opt-round-N/opt_<operator>.py` file for the final optimized source code.

Then, for each round that contributed to the final result, read that round's
`summary.md` or `attempts.md` to understand what changes were made and why.
Compare the operator source across rounds to identify the exact code regions
that were modified.

In the final source code shown in the report, annotate each key optimization
site with `# -- <comment> --` style inline comments that explain:

- Which round introduced the change
- The optimization theme or technique applied
- Why the change was made (performance bottleneck, correctness issue, etc.)
- What effect the change had (latency reduction, memory savings, etc.)

Place these annotation blocks directly above or adjacent to the modified code
regions. Each annotation block should reference the relevant round number so
the reader can cross-reference with the per-round detail sections.

### Step 5: Collect performance decomposition

Read the `Key optimization points` list from `opt-note.md` Overall Summary.
Each entry is in the form `<optimization point>: <improvement> (round N)`.

For latency data, read each round's `round-state.json` or `*_perf.txt` to
extract the geomean (average) latency.

### Step 6: Render report.md

Generate `report.md` following the format defined in `references/report-format.md`.
Write the completed report to `report.md` in the workspace root.

## Rules

- Write the report in **Chinese**.
- Use the exact format from `references/report-format.md` as the structure template.
- If the `Key optimization points` list is missing or incomplete from the Overall
  Summary of `opt-note.md`, display "未记录" (not recorded) in the performance
  decomposition section.
- If the hardware environment information is not provided in the prompt, note "未记录" for the verification info section.
- Do **not** modify any existing artifacts. Only write `report.md`.
- In the final code structure section, annotate the source code with `# -- <comment> --` inline comments that explain each optimization change, its origin round, reason, and effect. Do not just dump raw uncommented code.
- Keep the report professional and suitable for external review.
- Extract latency values from per-round benchmark output or `summary.md`; prefer
  geomean (average) values when available.
- Use `round-state.json` `correctness_status` field to determine whether each
  round's correctness validation passed.
