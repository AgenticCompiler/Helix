# CLI Help Environment Variables Design

## Summary

Extend `triton-agent --help` with a short environment variable section so users can discover the runtime knobs the CLI and bundled backends already honor.

## Goals

- Show the supported environment variables in top-level help.
- Keep command parsing and flag behavior unchanged.
- Keep the section concise and maintainable by deriving it from one internal table.

## Non-Goals

- Do not add new environment variables.
- Do not change per-command help pages.
- Do not document variables that are only used by external tooling.

## Proposed Behavior

- Top-level help adds an `Environment variables:` section after the command groups and examples.
- The section covers:
  - `TRITON_AGENT_BATCH_NPU_DEVICES`
  - `TRITON_AGENT_CODE_AGENT_MAX_RETRIES`
  - `TRITON_AGENT_BENCH_OUTPUT_DIR`
  - `TRITON_AGENT_COMPILER_SOURCE_CACHE_DIR`
  - `LLM_API_KEY`
  - `LLM_MODEL`
  - `LLM_BASE_URL`
- Each entry gets a one-line description in user-facing language.

## Implementation Notes

- Add one environment-variable metadata table in `src/triton_agent/cli.py`.
- Append a formatted environment-variable block to the existing top-level help epilog.
- Add parser tests that assert the section heading and variable names appear in `build_parser().format_help()`.
