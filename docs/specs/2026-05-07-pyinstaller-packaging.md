# PyInstaller Packaging

## User-Visible Semantics

`triton-agent` can be distributed as a PyInstaller `onedir` bundle that contains the Python CLI modules and the repository `skills/` tree. Users run the generated executable directly instead of invoking `uv run triton-agent`.

The packaged CLI still treats code-agent backends such as `codex`, `opencode`, `pi`, `claude`, and `traecli` as external tools that must be available on `PATH`. The package includes workflow skills and helper scripts, not third-party agent CLIs or the target Triton/Torch/NPU runtime.

Local execution of generated tests and benchmarks should use a normal Python interpreter from the user's environment. A packaged executable must not recursively launch itself when skill helpers need to execute generated Python files.

## Implementation

- Add a central `triton_agent.resources` module that resolves the application root from PyInstaller runtime metadata when frozen, and from the source repository root during normal development.
- Resolve `skills/` through that central resource helper instead of deriving repository parents from individual module locations.
- Keep `skills/` as plain data. The PyInstaller spec recursively includes non-cache files under `skills/` into the bundle at `skills/...`.
- Prefer `onedir` output for this project because the skill tree is large and should stay inspectable during packaging validation.
- Allow local generated-test and benchmark execution to use `TRITON_AGENT_PYTHON`; when unset in a frozen executable, fall back to `python`.
