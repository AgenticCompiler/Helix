---
name: helix-pyinstaller-packager
description: Package the helix repository into PyInstaller onefile executables with the built-in skills tree embedded. Use when Codex needs to build, repair, validate, or document Windows, Linux, or macOS helix release artifacts; create platform-tagged release archives; explain PyInstaller's per-OS build requirement; or verify that packaged helix can load bundled skills.
---

# Helix PyInstaller Packager

## Core Rule

PyInstaller is not a cross-compiler. Build Windows artifacts on Windows, Linux artifacts on Linux, and macOS artifacts on macOS. Do not claim that one host can produce native executables for every OS unless a separate VM/container/CI runner for each target OS is actually used.

## Expected Repository Shape

Use this skill for a `helix` repository that has:

- `pyproject.toml` managed by `uv`
- CLI entrypoint `helix.cli:main`
- source under `src/helix/`
- a top-level `skills/` directory that must be bundled as runtime data
- a PyInstaller build script at `scripts/build-pyinstaller.py`
- a PyInstaller spec at `packaging/helix.spec`; if it is missing, the build script can create a default one

## Packaging Mode

Build onefile executables by default. The release output should expose a single executable in each platform artifact directory, for example:

```text
dist/pyinstaller/helix-windows-x86_64/helix.exe
```

The built-in `skills/` tree must still be included as PyInstaller data. In onefile mode, do not expect a visible `_internal/skills` directory in the release output. PyInstaller extracts embedded data to a temporary `_MEI*` directory while the process is running, and `src/helix/resources.py` must resolve frozen resources through `sys._MEIPASS`.

This mode reduces source exposure in the distributed artifact, but it is not encryption or a hard reverse-engineering boundary.

## Workflow

1. Inspect the repository before editing:
   - `pyproject.toml`
   - `packaging/helix.spec`
   - `src/helix/resources.py`
   - current resource path usage for `skills/`
2. Ensure `pyinstaller` is available through `uv`:
   - If missing, add it as a dev dependency with `uv add --dev pyinstaller`.
3. Ensure runtime resource lookup supports both source and frozen execution:
   - Source mode should resolve the repository root.
   - Frozen mode should resolve PyInstaller bundle resources.
   - All runtime access to bundled `skills/` should flow through one resource helper.
4. Ensure the spec recursively includes `skills/` as data:
   - If `packaging/helix.spec` is missing, let the build script create its default spec or create an equivalent spec before building.
   - Include `SKILL.md`, `references/`, `scripts/`, `.json`, and other non-cache skill files.
   - Exclude `__pycache__` and `.pyc`.
   - Build a onefile `EXE` by passing `a.binaries` and `a.datas` into `EXE`; do not create a `COLLECT` onedir bundle for the default release path.
5. Build the current OS executable with the repository build script:
   - `uv run python scripts/build-pyinstaller.py --clean`
6. Validate the artifact:
   - Run packaged `helix --help`.
   - Confirm the platform artifact directory contains the onefile executable.
   - Run one command that loads bundled skill resources, such as `compare-perf` with tiny temp perf files.
   - For agent-backed commands, use `--verbose` to confirm skill staging into `.codex/skills`, `.claude/skills`, or the selected backend directory.
7. Report produced executable and release archive paths, validation results, and remaining environment limitations.

## Build Script

Use the repository-owned `scripts/build-pyinstaller.py` for the normal build path. The skill does not keep a second packaging implementation; keeping one script avoids drift between manual packaging, CI packaging, and skill-guided packaging.

The script:

- detects the current OS and CPU architecture
- creates a default `packaging/helix.spec` when the requested spec is missing
- invokes `uv run pyinstaller`
- writes platform-tagged output under `dist/pyinstaller/`
- validates the expected onefile executable
- creates a release archive unless `--no-archive` is used: `.zip` on Windows, `.tar.gz` on Linux and macOS
- includes the onefile executable and `README.md` in the release directory and archive

Examples:

```bash
uv run python scripts/build-pyinstaller.py --clean
uv run python scripts/build-pyinstaller.py --clean --no-archive
uv run python scripts/build-pyinstaller.py --clean --platform-tag linux-aarch64
```

## Multi-OS Release Procedure

For a release that needs Windows, Linux, and macOS artifacts:

1. Run the same script on a Windows host.
2. Run the same script on a Linux host.
3. Run the same script on a macOS host.
4. Collect the generated release archives from `dist/pyinstaller/`.

Read `references/platform-builds.md` when explaining this limitation to users or designing CI jobs.

## Validation Notes

- A packaged CLI can be validated without NPU hardware by checking `--help`, bundled skills, and pure parser/comparison commands.
- Real `run-test`, `run-bench`, `verify`, and `optimize` validation requires a Python environment with the target dependencies, often including `torch`, `torch_npu`, `triton`, and Ascend/CANN runtime.
- A packaged executable does not include external agent CLIs. Users still need `codex`, `claude`, `opencode`, `pi`, or other selected backend tools available on `PATH`.
- If generated tests or benchmarks are executed by a frozen app, ensure local execution uses an external Python interpreter, typically through `HELIX_PYTHON`.
