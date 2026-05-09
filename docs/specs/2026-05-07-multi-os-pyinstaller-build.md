# Multi-OS PyInstaller Build Script

## User-Visible Semantics

`triton-agent` provides one repository script for creating a PyInstaller bundle on the current operating system. The same script can be run on Windows, Linux, and macOS, but each platform must build its own executable on that platform.

The script produces a platform-tagged zip archive under `dist/pyinstaller/` so release artifacts can be collected from different build hosts or CI jobs.

## Constraints

- PyInstaller is not a cross-compiler. A Windows host builds a Windows executable, a Linux host builds a Linux executable, and a macOS host builds a macOS executable.
- The existing `packaging/triton-agent.spec` remains the source of truth for bundled Python modules and the `skills/` data tree.
- The build script should stay thin: invoke PyInstaller, validate expected outputs, and package the onedir bundle.
- The packaged application still depends on external agent CLIs and the user's target Triton/Torch/NPU runtime.

## Artifact Layout

For a Windows x86_64 host, the default output is:

```text
dist/pyinstaller/triton-agent-windows-x86_64/
dist/pyinstaller/triton-agent-windows-x86_64.zip
```

Linux and macOS use the same pattern with `linux` or `macos` in the platform tag.
