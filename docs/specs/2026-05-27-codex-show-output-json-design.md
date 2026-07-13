# Codex Show-Output JSON Rendering Design

Codex non-interactive `--show-output` should render Codex native JSON stdout into a readable timeline instead of streaming raw text or raw JSON into `helix-logs/<command>.show-output.log`.

## User-Visible Behavior

- `codex exec` adds `--json` when either `--show-output` or `--log-tools` is enabled.
- `--show-output` without `--log-tools` streams readable Codex timeline text to the terminal and appends that same readable text to the show-output log.
- `--show-output` alone does not create `helix-logs/otel/<run-id>/trace.jsonl`.
- `--log-tools` keeps writing structured trace events from the same Codex JSON stream.
- Ordinary non-interactive Codex runs without `--show-output` or `--log-tools` keep their current stdout behavior and do not add `--json`.

## Implementation

The Codex backend owns Codex-native JSON handling:

- `src/helix/backends/codex.py` decides when to pass `--json` and when to install `CodexJsonOutputFilter`.
- `src/helix/backends/codex_trace.py` parses Codex JSONL, renders readable timeline blocks, and writes trace events only when a trace path exists.

The generic `show_output_log.py` wrapper remains backend-neutral. It still writes attempt markers, appends filtered stdout, and computes generic line-prefix counts.

## Validation

Focused tests should prove:

- `--show-output` adds `--json`.
- `--show-output` without `--log-tools` returns `CodexJsonOutputFilter`.
- `CodexJsonOutputFilter(None, ...)` renders readable output and does not create trace files.
- Codex `item.started` and `item.completed` events render `[tool:start]` and `[tool:end]` timeline blocks.
- Non-JSON stdout still passes through as a fallback.
