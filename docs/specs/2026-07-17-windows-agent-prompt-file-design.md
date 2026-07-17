# Windows Agent Prompt File

## User-visible behavior

On Windows, code-agent launches no longer put the complete task prompt in the
command line. Helix writes the complete prompt to the temporary UTF-8 file
`USER_PROMPT.md` in the target workspace, then launches the backend with a
short prompt that tells the agent to read and follow that file before doing any
work.

This applies consistently to command-line backends such as Codex, Claude,
OpenCode, Pi, and Trae CLI. It prevents a long optimize prompt from exceeding
the Windows command-line length limit, including when a backend executable is
wrapped by `cmd /c`.

Interactive sessions also receive the short file-reading instruction. The file
approach does not use standard input, so it remains compatible with terminal
UIs. The OpenHands backend is unchanged: it sends the prompt through its SDK
rather than a process command line.

The temporary prompt file exists for the complete launch, including automatic
backend retries, and is removed after the launch completes or raises an error.
Helix overwrites `USER_PROMPT.md` when preparing a run because the file is
runner-managed runtime state.

## Implementation shape

`AgentRunner.run` prepares a launch-only copy of every request on Windows. The
original prompt is written to `request.workdir / "USER_PROMPT.md"`; the launch
copy contains a short, absolute-path instruction to read that file. Backend
`build_command` methods remain unchanged, so the common mechanism covers every
process-backed backend.

The file intentionally does not live under `.helix/`: optimize hooks deny
agent reads of runner-managed files there, whereas the agent must be allowed to
read this prompt file. The base runner owns cleanup so retries can reuse the
same file without exposing an artifact after completion.

## Verification

- A Windows-simulated base-runner test verifies the backend command receives a
  short file-reading prompt and that the file contains the full task prompt for
  a convert request.
- The same test verifies cleanup after the subprocess returns.
