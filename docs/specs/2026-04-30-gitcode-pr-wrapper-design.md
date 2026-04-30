# GitCode PR UV Wrapper Design

## Summary

Add a repo-local shell wrapper for the `managing-gitcode-prs` skill so the skill can run `gc pr` through `uv` without turning GitCode CLI into a repository-wide development dependency.

## Motivation

The skill now has a stable default repository target and a confirmed `gc` command entrypoint backed by a wheel URL. The remaining gap is execution guidance: asking agents to call `gc` directly assumes the CLI is already installed on the host, while adding the package to `[dependency-groups].dev` would make a repo-local PR helper look like a core development dependency of `triton-agent`.

## Decision

- Keep GitCode CLI out of `pyproject.toml` dependencies and dev dependencies.
- Add `.codex/skills/managing-gitcode-prs/scripts/run-gc-pr.sh`.
- Make the wrapper:
  - require `GC_TOKEN`
  - default to the provided wheel URL for `gitcode_cli-0.3.11`
  - allow overriding the wheel URL with `GITCODE_CLI_WHEEL_URL`
  - set a writable `UV_CACHE_DIR` when unset
  - execute `uv tool run --from <wheel-url> gc pr ...`
- Update the skill docs so `SKILL.md` and the reference file prefer the wrapper over a bare `gc pr` call.

## Non-Goals

- Do not add a new project CLI command under `src/`.
- Do not add GitCode CLI to `[dependency-groups].dev`.
- Do not broaden the skill beyond pull request flows.

## Verification

- Add a contract test that requires:
  - the wrapper script to exist
  - the skill docs to mention the wrapper
  - the wrapper to reference `uv tool run --from`, `gc pr`, and `GC_TOKEN`
- Run the targeted contract test.
- Run `bash -n` on the wrapper for shell syntax validation.
- Re-run the skill validator after the doc updates.
