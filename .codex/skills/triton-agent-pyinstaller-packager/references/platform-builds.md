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
triton-agent-windows-x86_64.zip
triton-agent-linux-x86_64.zip
triton-agent-linux-aarch64.zip
triton-agent-macos-x86_64.zip
triton-agent-macos-aarch64.zip
```

The zip should contain the whole PyInstaller onedir bundle. Users must not copy only the executable because bundled Python libraries and `_internal/skills` are required at runtime.

## Minimum Validation

If `packaging/triton-agent.spec` is missing, use the bundled script's default-spec creation path. The generated spec should be committed or reviewed before release because it defines which resources are bundled.

Run these checks on every target OS after building:

```bash
<bundle>/triton-agent --help
```

On Windows:

```powershell
<bundle>\triton-agent.exe --help
```

Confirm bundled skills:

```text
<bundle>/_internal/skills
```

Run a pure command that loads bundled skill scripts, for example `compare-perf` with temporary files:

```text
latency-case-1: 10
latency-case-1: 8
```

Then invoke:

```bash
triton-agent compare-perf --baseline baseline.txt --compare compare.txt
```

The expected success marker is:

```text
PASS: compared 1 latency entries
```

## Runtime Dependencies Not Bundled

The package intentionally does not bundle:

- `codex`, `claude`, `opencode`, `pi`, or other external agent CLIs
- target operator runtime environments
- NPU drivers, CANN, `torch_npu`, or Ascend hardware access

Document those as host prerequisites, not packaging failures.
