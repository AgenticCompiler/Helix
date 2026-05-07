---
name: triton-agent-pyinstaller-packager
description: Package the triton-agent repository into PyInstaller onedir executable bundles with the built-in skills tree included. Use when Codex needs to build, repair, validate, or document Windows, Linux, or macOS triton-agent release artifacts; create platform-tagged zip archives; explain PyInstaller's per-OS build requirement; or verify that packaged triton-agent can load bundled skills.
---

# Triton Agent PyInstaller Packager

## Core Rule

PyInstaller is not a cross-compiler. Build Windows artifacts on Windows, Linux artifacts on Linux, and macOS artifacts on macOS. Do not claim that one host can produce native executables for every OS unless a separate VM/container/CI runner for each target OS is actually used.

## Expected Repository Shape

Use this skill for a `triton-agent` repository that has:

- `pyproject.toml` managed by `uv`
- CLI entrypoint `triton_agent.cli:main`
- source under `src/triton_agent/`
- a top-level `skills/` directory that must be bundled as runtime data
- a PyInstaller spec at `packaging/triton-agent.spec`; if it is missing, the bundled script can create a default one

## Workflow

1. Inspect the repository before editing:
   - `pyproject.toml`
   - `packaging/triton-agent.spec`
   - `src/triton_agent/resources.py`
   - current resource path usage for `skills/`
2. Ensure `pyinstaller` is available through `uv`:
   - If missing, add it as a dev dependency with `uv add --dev pyinstaller`.
3. Ensure runtime resource lookup supports both source and frozen execution:
   - Source mode should resolve the repository root.
   - Frozen mode should resolve PyInstaller bundle resources.
   - All runtime access to bundled `skills/` should flow through one resource helper.
4. Ensure the spec recursively includes `skills/` as data:
   - If `packaging/triton-agent.spec` is missing, let the build script create its default spec or create an equivalent spec before building.
   - Include `SKILL.md`, `references/`, `scripts/`, `.json`, and other non-cache skill files.
   - Exclude `__pycache__` and `.pyc`.
5. Build the current OS bundle with the bundled script:
   - `python <this-skill>/scripts/package_triton_agent.py --repo <repo> --clean`
6. Validate the artifact:
   - Run packaged `triton-agent --help`.
   - Confirm bundled `_internal/skills` exists and contains skill directories.
   - Run one command that loads a bundled skill script, such as `compare-perf` with tiny temp perf files.
   - For agent-backed commands, use `--verbose` to confirm skill staging into `.codex/skills`, `.claude/skills`, or the selected backend directory.
7. Report produced bundle and zip paths, validation results, and remaining environment limitations.

## Bundled Script

Use `scripts/package_triton_agent.py` for the normal build path. It:

- detects the current OS and CPU architecture
- creates a default `packaging/triton-agent.spec` when the requested spec is missing
- invokes `uv run pyinstaller`
- writes platform-tagged output under `dist/pyinstaller/`
- validates the expected executable and bundled skills directory
- creates a zip archive unless `--no-zip` is used

Examples:

```bash
python path/to/skill/scripts/package_triton_agent.py --repo /path/to/triton-agent --clean
python path/to/skill/scripts/package_triton_agent.py --repo . --clean --no-zip
python path/to/skill/scripts/package_triton_agent.py --repo . --platform-tag linux-aarch64
```

## Multi-OS Release Procedure

For a release that needs Windows, Linux, and macOS artifacts:

1. Run the same script on a Windows host.
2. Run the same script on a Linux host.
3. Run the same script on a macOS host.
4. Collect the generated zip files from `dist/pyinstaller/`.

Read `references/platform-builds.md` when explaining this limitation to users or designing CI jobs.

## Validation Notes

- A packaged CLI can be validated without NPU hardware by checking `--help`, bundled skills, and pure parser/comparison commands.
- Real `run-test`, `run-bench`, `verify`, and `optimize` validation requires a Python environment with the target dependencies, often including `torch`, `torch_npu`, `triton`, and Ascend/CANN runtime.
- A packaged executable does not include external agent CLIs. Users still need `codex`, `claude`, `opencode`, `pi`, or other selected backend tools available on `PATH`.
- If generated tests or benchmarks are executed by a frozen app, ensure local execution uses an external Python interpreter, typically through `TRITON_AGENT_PYTHON`.
