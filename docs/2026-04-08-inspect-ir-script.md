# Inspect IR Script For Archived Ascend Operator IR

## Goal

Add a bundled inspection script for `skills/ascend-operator-ir-analyzer/` so agents can navigate and compare large archived IR captures without manually opening dozens of `.mlir` files. The script should help an agent answer three early performance-debugging questions:

- which stages are worth inspecting
- what a given stage roughly contains
- what changed between two stages

## User-visible behavior

- Add a new script at `skills/ascend-operator-ir-analyzer/scripts/inspect_ir.py`.
- The script should expose one entrypoint with subcommands instead of multiple independent scripts.
- All subcommands should accept `--ir-dir <path>` and resolve `bishengir_stages/` internally.
- The script should not expose `bishengir_stages/` as a user-facing argument because that is an archive implementation detail.
- The initial subcommands should be:
  - `list-stages`
  - `stage-summary`
  - `diff-stages`
- The initial output format should be terminal-oriented plain text only.

## Subcommand contract

### `list-stages`

Purpose:
- quickly scan available stages
- filter by pass name or other keywords

Input:
- `--ir-dir <path>`
- optional `--grep <pattern>`
- optional `--limit <N>`

Output:
- one stage per line
- include the stage filename stem and the relative path under `bishengir_stages/`
- include a small size hint so large stages stand out

### `stage-summary`

Purpose:
- quickly decide whether a stage is worth deeper inspection

Input:
- `--ir-dir <path>`
- `--stage <selector>`

Selector rules:
- support exact relative path under `bishengir_stages/`
- support exact filename stem such as `41_hivm-plan-memory`
- support a unique substring match such as `hivm-plan-memory`
- fail explicitly on zero or multiple matches

Output:
- a short stable text summary with sections such as:
  - stage identity
  - file size and line count
  - keyword counts
  - highlights

The first version should keep keyword analysis lightweight and heuristic-based. Useful counters include:
- `alloc`
- `copy`
- `matmul`
- `for`
- `if`
- `vector`
- `load`
- `store`
- `dma`
- `wait`
- `set_flag`
- `barrier`

### `diff-stages`

Purpose:
- see what changed across two selected stages

Input:
- `--ir-dir <path>`
- `--from <selector>`
- `--to <selector>`
- optional `--context <N>`

Output:
- a short header showing both selected stages
- a compact summary of line count, file size, and keyword-count deltas
- a unified diff with configurable context

## Design notes

- Keep the first version simple and text-oriented. The immediate problem is IR volume, not downstream automation format.
- Structure the code so `--format json` can be added later without redesigning stage resolution.
- Reuse shared helpers inside the script for:
  - archive validation
  - stage discovery
  - stage selector resolution
  - keyword counting
- Prefer readable stable output over dense raw dumps. Agents should be able to scan the output and decide the next file to inspect.
- Do not add opinionated performance diagnosis rules yet. This script is for navigation and comparison, not automatic root-cause claims.

## Documentation updates

- Update `skills/ascend-operator-ir-analyzer/SKILL.md` so the skill tells agents to use `inspect_ir.py` for navigation, summary, and diff once IR capture is complete.
- Update `README.md` and `AGENTS.md` so repository-level expectations mention the new inspection helper as part of the IR-analysis workflow.

## Verification

- Add unit tests for:
  - stage discovery and filtering
  - stage selector resolution
  - stage summary output
  - diff output
- Run targeted tests for the new script.
- Run the repository unittest suite, `pyright`, and relevant `ruff` checks.
