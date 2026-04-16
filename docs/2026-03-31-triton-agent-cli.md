# Triton Agent CLI for Test, Benchmark, and Optimization

> **Superseded note:** This early design predates the current repository skill naming. The active skill mapping now uses `triton-npu-gen-test`, `triton-npu-gen-bench`, `triton-npu-gen-eval-suite`, `triton-npu-run-eval`, `triton-npu-optimize`, and `triton-npu-optimize-check`.

## Summary

Build a `uv`-managed Python CLI project that wraps code agents behind a small abstraction layer and maps its kebab-case subcommands to the existing local skills:

- `gen-test` -> `triton-npu-gen-test`
- `gen-eval` -> `triton-npu-gen-eval-suite`
- `gen-bench` -> `triton-npu-gen-bench`
- `run-test` / `run-bench` / `compare-*` -> `triton-npu-run-eval`
- `optimize` -> `triton-npu-optimize`

The implementation should use `argparse`, support `--input/-i`, `--output/-o`, `--interact`, `--verbose`, `--show-output`, and `--agent` on all subcommands, plus `--force-overwrite` on generation commands, and start with `codex` as the only concrete agent backend. The design should keep agent-specific preparation and cleanup isolated so other backends can be added later.

## Key Changes

### Project structure and packaging

- Initialize a `uv` Python project with a console entrypoint such as `triton-agent`.
- Keep the code in a small package with focused modules for:
  - CLI parsing and subcommand dispatch
  - agent abstraction and shared request/result types
  - Codex-specific runner
  - skill staging preparation/cleanup
  - optimize watchdog/supervisor
- Use English for all prompts, comments, log messages, and internal docs.

### CLI behavior

- Implement five kebab-case subcommands: `gen-test`, `run-test`, `gen-bench`, `run-bench`, `optimize`.
- `gen-test`, `gen-bench`, and `optimize` accept:
  - `--input/-i`: required path to the operator file
  - `--output/-o`: optional output artifact path
  - `--interact`: when set, attach to the live agent TUI/session instead of background-style non-interactive execution
  - `--verbose`: print the concrete code-agent launch command before execution and show skill-link creation and cleanup messages
  - `--show-output`: in non-interactive mode, stream the readable code-agent output to the current terminal while still collecting it for result handling
  - `--agent`: backend selector, default `codex`
- `run-test` accepts:
  - `--test-file`: required generated test file path
  - `--operator-file`: required operator file path
  - `--output/-o`, `--interact`, `--verbose`, `--show-output`, `--agent`, `--test-mode`
- `run-bench` accepts:
  - `--bench-file`: required generated benchmark file path
  - `--operator-file`: required operator file path
  - `--output/-o`, `--interact`, `--verbose`, `--show-output`, `--agent`, `--bench-mode`
- `gen-test` and `gen-bench` also accept:
  - `--force-overwrite`: allow replacing an existing generated output file; otherwise the CLI refuses to overwrite
- For `run-test` and `run-bench`, validate the explicit harness path and explicit operator path directly instead of deriving a generated artifact path by convention.
- Use consistent default output inference for generated files when `--output` is omitted, with names based on the operator stem and command type.
- Keep an internal explicit mapping from CLI command name to skill name even where the names are similar, so dispatch stays deterministic and easy to extend.

### Agent abstraction

- Define a backend-neutral `AgentRunner` interface with clear stages:
  - pre-run preparation
  - command/session launch
  - optional interactive attach
  - result collection
  - post-run cleanup
- Define a request model that includes:
  - subcommand kind
  - input path
  - optional output path
  - interactive flag
  - resolved skill name
  - generated prompt text
- Implement `CodexRunner` as the first backend:
  - non-interactive mode uses `codex exec` with normal readable text output
  - interactive mode uses `codex` TUI-style launch in the target working directory
- Keep prompt construction centralized so each subcommand has a deterministic English instruction template that tells the agent which local skill to use and what files to read/write.

### Skill staging lifecycle

- Before launching the agent, ensure the working directory exposes this repo’s local skills in the backend-specific expected location.
- For Codex:
  - target path is `.codex/skills` under the agent working directory
  - if `.codex/skills` does not exist, create the parent directory if needed and copy this repo’s `skills/` tree into `.codex/skills`
  - if `.codex/skills` already exists, copy only missing per-skill directories from this repo’s `skills/`
- Track which copied paths were created by this run so cleanup deletes only owned staged content.
- Handle partial failure safely:
  - creation failures should abort with a clear error
  - cleanup failures should be reported but should not hide the main command result
  - never delete pre-existing non-staged content

### Optimize supervision

- Implement a dedicated `OptimizeSupervisor` around the agent runner for long-running `optimize`.
- Default recovery strategy: auto-resume.
  - Detect stalls using inactivity timeout plus optional repeated-output/no-progress heuristics
  - first try to continue the current session by injecting a short continuation prompt
  - if that fails or the process is dead, relaunch the agent with a compact progress summary and the original task context
- Persist enough run metadata locally to support resume/restart decisions within the same optimization job:
  - original request
  - last known session/process metadata if available
  - recent output snippet or synthesized progress summary
  - recovery attempt count
- Put hard limits on automatic recovery attempts and surface a clear failure when exhausted.
- In interactive mode, watchdog behavior should be conservative:
  - allow attach to the live session
  - avoid fighting the user for control
  - only warn or recover when the session is clearly stalled/disconnected

## Public Interfaces and Defaults

- CLI executable: one top-level command with five subcommands.
- Initial backend enum/list should expose `codex`, while keeping the interface open for later providers.
- Prompt templates should explicitly instruct the agent to use one of the local skills and operate on the supplied operator file.
- Default decisions locked in:
  - CLI framework: `argparse`
  - subcommand naming: kebab-case
  - `run-test` input semantics: explicit test file plus explicit operator file
  - `run-bench` input semantics: explicit benchmark file plus explicit operator file
  - `--interact`: attach to live agent UI/session
  - `--verbose`: log the launch command and skill-link lifecycle without changing execution behavior
  - `--show-output`: stream readable non-interactive agent output without changing exit-code or aggregation behavior
  - `--force-overwrite`: only affects `gen-test` and `gen-bench`; default behavior is to protect existing output files
  - optimize recovery: auto-resume first, restart second
  - architecture: structured core, not thin wrapper and not plugin-heavy v1

## Test Plan

- Unit tests for argument parsing and subcommand-to-skill mapping.
- Unit tests for run-command path validation and missing-file errors for `run-test` and `run-bench`.
- Unit tests for skill staging behavior:
  - missing `.codex/skills`
  - existing `.codex/skills`
  - mixed pre-existing content
  - cleanup of only owned copied paths
- Unit tests for prompt generation to ensure the correct skill and file arguments are included.
- Unit tests for `CodexRunner` command construction in interactive and non-interactive modes.
- Unit tests for `OptimizeSupervisor` stall detection and recovery policy.
- Integration-style tests with a fake/mock agent runner to verify end-to-end dispatch without requiring real Codex execution.
- Optional smoke test gated by environment if `codex` is installed, validating basic launch command assembly without asserting real model output.

## Assumptions

- The existing `skills/` directory is the source of truth and skill contents are already good enough for v1.
- v1 only needs one real backend implementation: `codex`.
- The agent working directory is the operator file’s directory unless a later requirement introduces a separate workspace rule.
- `--output` is accepted on all subcommands for interface consistency, even if run commands may primarily use it for explicit artifact/result locations rather than source generation.
- Optimize supervision is process/session-level orchestration only; it does not need deep semantic understanding of the operator beyond summarizing progress and re-prompting.
