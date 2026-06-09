# Pytest Verification Entrypoint Design

## Summary

Align the root repository verification entrypoint with the durable `AGENTS.md` guidance by adding `pytest` to the root development dependencies and updating the `README.md` verification section to use the agent-friendly low-noise `pytest` command.

## Motivation

The repository currently describes two different top-level test entrypoints: root docs still point to `unittest discover`, while `AGENTS.md` now defines a single low-noise `pytest` command as the standard repository verification step. That split makes the workflow ambiguous for both humans and agents.

## User-Visible Semantics

- Root repository verification continues to consist of three commands:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
- The root `dev` dependency group explicitly includes `pytest` so the documented verification command is available from a fresh development environment.
- This change does not alter service-specific test stacks or broaden the repository-wide verification scope beyond the existing root `tests/` suite.

## Scope Boundaries

- In scope:
  - root `pyproject.toml`
  - `README.md` verification command wording
- Out of scope:
  - migrating all historical docs or plans
  - changing service-local test dependencies
  - changing test implementation style across the repository

## Verification

- Confirm root `pyproject.toml` includes `pytest` in `[dependency-groups].dev`.
- Confirm `README.md` verification section matches the durable `AGENTS.md` command.
