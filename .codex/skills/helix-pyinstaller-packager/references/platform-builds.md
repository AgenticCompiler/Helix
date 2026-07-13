# Platform Builds

## PyInstaller Platform Boundary

PyInstaller produces executables for the operating system it runs on:

- Windows host -> Windows `.exe`
- Linux host -> Linux executable
- macOS host -> macOS executable or app bundle shape depending on spec

Do not describe this as cross-compilation. If the user asks for all OS artifacts, plan separate target runners:

- local Windows machine, Windows VM, or Windows CI runner
- Linux machine, Linux VM/container, or Linux CI runner
- macOS machine or macOS CI runner

## Artifact Naming

Use platform tags that include OS and architecture:

```text
helix-windows-x86_64.zip
helix-linux-x86_64.tar.gz
helix-linux-aarch64.tar.gz
helix-macos-x86_64.tar.gz
helix-macos-aarch64.tar.gz
```

Use `.zip` for Windows artifacts. Use `.tar.gz` for Linux and macOS artifacts so executable mode bits are preserved when users extract the archive on Unix-like systems.

The release archive should contain the platform-tagged release directory with the onefile executable and README inside it. For example:

```text
helix-windows-x86_64/
  helix.exe
  README.md
```

On Linux:

```text
helix-linux-aarch64-release/
  helix
  README.md
```

The built-in `skills/` tree is embedded into the executable as PyInstaller data. Do not expect a visible `_internal/skills` directory in the release artifact. PyInstaller extracts embedded data to a temporary `_MEI*` directory while the process is running.

## Minimum Validation

If `packaging/helix.spec` is missing, use the repository build script's default-spec creation path. The generated spec should be committed or reviewed before release because it defines which resources are bundled.

Run these checks on every target OS after building:

```bash
<artifact-dir>/helix --help
```

On Windows:

```powershell
<artifact-dir>\helix.exe --help
```

Confirm bundled skills through behavior instead of by checking a visible `_internal` directory. Run a pure command that loads bundled skill scripts, for example `compare-perf` with temporary files:

```text
latency-case-1: 10
latency-case-1: 8
```

Then invoke:

```bash
helix compare-perf --baseline baseline.txt --compare compare.txt
```

The expected success marker is:

```text
PASS: compared 1 latency entries
```

For agent-backed commands, also run with `--verbose` and confirm skills are staged into the selected backend's workspace skill directory.

## Source Exposure Note

Onefile packaging hides `skills/` from the visible release directory, which matches the intended distribution shape when skill sources should not be published as plain files next to the executable. It is not a strong protection mechanism: PyInstaller must extract embedded files to a temporary runtime directory, and determined users can still inspect packaged content.

## Runtime Dependencies Not Bundled

The package intentionally does not bundle:

- `codex`, `claude`, `opencode`, `pi`, or other external agent CLIs
- target operator runtime environments
- NPU drivers, CANN, `torch_npu`, or Ascend hardware access

Document those as host prerequisites, not packaging failures.
