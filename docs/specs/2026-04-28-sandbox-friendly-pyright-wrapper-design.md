# Sandbox-Friendly Pyright Wrapper Design

## Summary

Developers frequently need an additional file-scoped strict `pyright` check for Python files under `skills/*/scripts/`. Running the current ad hoc `uv run pyright` command often requires sandbox escalation because `uv` defaults to a cache directory outside the writable sandbox. This change adds a repository-owned wrapper script that pins `UV_CACHE_DIR` to a writable temporary directory and standardizes the strict file-scoped check contract in `AGENTS.md`.

## Goal

- Provide a repo-local command for strict `pyright` checks on skill scripts that works inside the sandbox by default.
- Remove the need to remember the long temporary-`pyproject.toml` command.
- Make `AGENTS.md` point contributors to the wrapper script for the required strict skill-script check.

## Decision

- Add a shell script under `scripts/` dedicated to strict `pyright` checks for skill scripts.
- The script should:
  - set `UV_CACHE_DIR` to a writable sandbox-friendly directory when the variable is not already set
  - create a temporary `pyproject.toml` with `pythonVersion = "3.11"` and `typeCheckingMode = "strict"`
  - add each target file's parent directory to `extraPaths`
  - run `uv run pyright --project <temp-config> <targets...>` from the repository root
- Update `AGENTS.md` so the stable project rule explicitly prefers this wrapper for `skills/*/scripts/` strict checks.

## Non-Goals

- Do not change the repository-wide default `pyright` configuration in `pyproject.toml`.
- Do not promote all skill scripts to strict mode in the default `uv run pyright`.
- Do not add new CLI surface under `src/`.

## Verification

- Add a contract test that asserts:
  - the wrapper script exists
  - the wrapper script configures `UV_CACHE_DIR`
  - `AGENTS.md` points skill-script strict checks to the wrapper script
- Run the targeted contract test.
- Run the wrapper itself against `skills/triton-npu-run-eval/scripts/bench_runner.py`.
