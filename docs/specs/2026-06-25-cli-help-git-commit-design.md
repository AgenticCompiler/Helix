# CLI Help Git Commit Design

## Summary

Extend `triton-agent --help` so the top-level help output includes the CLI build's Git commit.

The design must support four runtime shapes with one consistent user-facing rule:

- source checkout execution such as `uv run triton-agent --help`
- editable installs such as `pip install -e .`
- ordinary installed wheels
- PyInstaller onefile executables

## Goals

- Show a Git commit identifier in top-level `triton-agent --help`.
- Keep editable installs aligned with the current checkout `HEAD`, not the install-time commit.
- Keep ordinary wheel installs and PyInstaller binaries stable by embedding build metadata at build time.
- Keep the CLI thin by resolving build metadata in a dedicated helper instead of spreading Git and packaging logic through `cli.py`.

## Non-Goals

- Do not change subcommand help pages.
- Do not add a new public `version` subcommand or `--version` flag in this change.
- Do not change the package version string in `pyproject.toml`.
- Do not fail command execution when commit metadata is unavailable.
- Do not surface dirty-working-tree state in this change.

## User-Visible Semantics

- Top-level help adds a short `Build info:` section.
- The section includes one line:
  - `Git commit: <value>`
- `triton-agent -h` and `triton-agent --help` render the same build-info output.
- The stored value is a full 40-character SHA when available.
- The help output renders a shortened 12-character SHA for readability.
- When no commit can be resolved, help still renders the section and prints `Git commit: unknown`.

## Resolution Order

Commit resolution follows this order:

1. Resolve the current checkout `HEAD` when the running code is backed by a Git checkout.
2. Otherwise read embedded build metadata that was generated at build time.
3. If neither source succeeds, report `unknown`.

This resolution happens eagerly when `build_parser()` constructs the top-level parser, because the current CLI passes `_build_top_level_epilog()` directly into `ArgumentParser(...)`. The resolver must therefore stay lightweight, best-effort, and safe to call even when the user is not about to print help. Process-local memoization is acceptable.

This order gives the intended semantics for each runtime shape:

- source checkout: show the current repository `HEAD`
- editable install: show the current repository `HEAD`
- ordinary wheel: show the commit recorded when the wheel was built
- PyInstaller onefile: show the commit recorded when the executable was built

## Source Checkout Semantics

Source-backed execution includes both direct repository runs and editable installs, because both execute code from the repository tree.

The resolver should treat a runtime as source-backed when it can discover a repository root for the loaded package and that root contains Git metadata. The detection must support both:

- a `.git` directory
- a `.git` indirection file used by Git worktrees

The resolver should then read the checkout `HEAD` from that repository state. The implementation may shell out to Git or read repository metadata directly, but the helper must treat failure as non-fatal and continue to the embedded metadata fallback.

## Embedded Build Metadata

Non-source distributions need a build-time metadata source that survives installation and freezing.

The implementation should define one embedded metadata contract with a single source of truth, for example a small JSON payload that records:

- `git_commit`

The embedded file path should be package-relative and stable across wheel and frozen layouts:

- `triton_agent/_build_meta.json`

The runtime helper should load only this one embedded metadata source for non-source installs. The design should not duplicate commit values across multiple files or hard-coded constants.

## Build Integration

The build pipeline should materialize embedded metadata without dirtying the source tree.

### Wheel And Sdist Builds

- Because the repository already uses `setuptools.build_meta`, metadata generation should hook into the setuptools build flow rather than rely on a separate wrapper script.
- A small setuptools command override is the preferred design, so ordinary PEP 517 build callers still produce metadata automatically.
- The command override should be registered through the existing setuptools configuration path, for example `pyproject.toml` `tool.setuptools.cmdclass`, so this change does not need to introduce a new `setup.py`.
- The build hook should accept a build-time environment override such as `TRITON_AGENT_BUILD_GIT_COMMIT`.
- When that environment variable is set, it is the authoritative build commit source.
- When the environment variable is absent, the hook may fall back to resolving `git rev-parse HEAD` from the source checkout.
- If both the environment override and Git lookup are unavailable, for example in shallow or source-stripped CI inputs, the hook should skip commit metadata generation rather than fail the build.
- Build-time metadata should be generated into build output or staged archive content, not committed source files.
- `build_py` should stage `triton_agent/_build_meta.json` into the built package output.
- The sdist flow does not need to ship a pre-generated `triton_agent/_build_meta.json` in the unpacked archive as long as downstream wheel builds can regenerate or preserve the same package-relative metadata through the packaged build hook.
- The generated metadata must be included in the installed package so ordinary wheel installs can resolve the recorded commit without repository access.

### PyInstaller Builds

- The PyInstaller packaging flow should include the same `triton_agent/_build_meta.json` payload in the bundled application data.
- `packaging/triton-agent.spec` should therefore add that package-relative metadata file to `a.datas`, alongside the existing bundled skills data.
- The onefile executable must therefore report the build commit even though the frozen runtime is no longer backed by a Git checkout.

### Editable Installs

- Editable installs should not pin commit metadata at install time.
- Because editable installs execute from the repository checkout, they should continue to resolve the live checkout `HEAD` each time help is rendered.

## CLI Integration

- Add one narrow runtime helper module such as `src/triton_agent/build_info.py`.
- The helper should separate raw resolution from display formatting:
  - one function returns the full resolved commit or `None`
  - one function returns the display value used by help output, including 12-character shortening and `unknown` fallback
- `cli.py` should consume a formatted build-info line or section from that helper.
- The top-level help epilog should append the `Build info:` section near the other top-level reference blocks.
- Existing parsing behavior, command registration, and subcommand help text should remain unchanged.

## Metadata Lookup Path

- Embedded metadata lookup should be package-relative, for example by resolving `Path(__file__).with_name("_build_meta.json")` from `build_info.py`.
- This package-relative contract works for installed wheels and for PyInstaller when the bundled data is placed under the `triton_agent/` package path.
- The existing frozen-runtime pattern in `src/triton_agent/resources.py` is still a useful precedent, but the build-info lookup should not depend on `application_root()` alone because installed wheels need package-relative, not repository-root, resolution semantics.

## Failure Handling

- Missing or invalid Git metadata must not make `triton-agent --help` fail.
- Missing or invalid embedded metadata must not make `triton-agent --help` fail.
- Invalid metadata should be ignored and treated the same as an unavailable commit.
- The final fallback is always `Git commit: unknown`.

## Verification

- Add focused unit tests for the build-info resolver covering:
  - source checkout resolution
  - embedded metadata resolution
  - unknown fallback
- Extend the top-level CLI help tests to assert that `Build info:` and `Git commit:` appear in `build_parser().format_help()`.
- Validate editable-install semantics by confirming the resolver prefers source checkout state over embedded metadata when both are present.
- Validate a non-editable packaged path by confirming an installed wheel can report an embedded commit without repository access.
- Treat frozen-binary verification as an integration test, not a fast unit test.
- Validate a frozen path by confirming a PyInstaller-built executable reports an embedded commit.
