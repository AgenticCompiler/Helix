# PyInstaller Onefile Packaging

## Goal

Package `triton-agent` as a single PyInstaller executable per target OS so the `skills/` tree is embedded in the binary artifact instead of being exposed as a visible `_internal/skills` directory in the release output.

## User-Visible Semantics

- The default PyInstaller release artifact is now a onefile executable.
- The executable still contains the built-in `skills/` runtime data.
- Users run the executable directly from the platform artifact directory or from the generated release archive.
- Release archives include the onefile executable plus `README.md`.
- Windows release archives use `.zip`; Linux and macOS release archives use `.tar.gz` so executable permissions survive extraction on Unix-like systems.
- PyInstaller still extracts bundled files to a temporary `_MEI*` runtime directory while the process is running. This packaging mode reduces release artifact exposure, but it is not encryption or a hard reverse-engineering barrier.
- PyInstaller remains platform-local: build Windows binaries on Windows, Linux binaries on Linux, and macOS binaries on macOS.

## Implementation

- `packaging/triton-agent.spec` builds a onefile `EXE` and passes `a.binaries` and `a.datas` directly into the executable instead of creating a `COLLECT` onedir bundle.
- `scripts/build-pyinstaller.py` expects the executable at the platform artifact root, for example `dist/pyinstaller/triton-agent-windows-x86_64/triton-agent.exe`.
- The PyInstaller packager skill delegates to `scripts/build-pyinstaller.py` instead of keeping a second packaging implementation.
- The scripts no longer validate a visible `_internal/skills` directory because onefile artifacts do not expose that directory in the release output.
- Generated release archives contain only the onefile executable and `README.md` under the platform-tagged release directory. The archive writer must not recursively include stale files from an old onedir build directory.

## Validation

Minimum validation after building:

```bash
triton-agent --help
```

Then run a command that requires bundled skill resources, such as a small `compare-perf` check. This validates that `skills/` can be loaded from the PyInstaller runtime extraction directory.
