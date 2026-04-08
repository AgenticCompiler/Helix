# Ascend Operator IR Analyzer Skill

## Goal

Add a new repository-owned skill that helps a code agent capture complete Triton Ascend compilation IR for an operator workflow, archive the artifacts in a stable layout, and analyze potential performance issues from the archived IR. The skill should also tell the agent when to pair IR inspection with the existing Ascend NPU profiler skill for hotspot-backed diagnosis.

## User-visible behavior

- Add a new skill named `ascend-operator-ir-analyzer` under `skills/`.
- The skill should trigger when the user wants to inspect Triton Ascend operator IR, archive compiler stages, or analyze potential performance issues from generated IR artifacts.
- The default workflow should start by running a bundled script that executes an operator-related command with:
  - `TRITON_DEBUG=1`
  - `TRITON_ALWAYS_COMPILE=1`
- The script should accept:
  - an IR directory path
  - `--bench-file <path>`
  - `--operator-file <path>`
  - optional remote settings matching the repository's existing remote execution semantics
- The script should construct the benchmark command itself instead of accepting an arbitrary trailing command, using:
  - `python3 <bench-file-name> --operator-file <operator-file-path>`
- From the command stdout, the script should extract:
  - the dumped Triton intermediate directory from a line that starts with `Dumping intermediate results to `
  - the Bisheng IR compile command from a line that starts with `[DEBUG] cmd_list: `
- The script should normalize the extracted compile command so `--append-bisheng-options=...` preserves embedded spaces as one argument when the command is replayed.
- The script should copy the dumped Triton IR directory into the IR directory before replaying the compiler.
- The script should replay the Bisheng compile command against the archived `kernel.ttadapter.mlir`, remove one-shot print filters such as `--bishengir-print-ir-after=...`, add `--mlir-print-ir-after-all`, add `--mlir-print-ir-tree-dir=<ir-dir>/bishengir_stages`, and redirect stderr to `<ir-dir>/all-ir.txt`.
- After capture completes, the IR directory should contain enough metadata for later analysis without re-running the benchmark command.
- The skill should instruct the code agent to inspect the archived IR and analyze likely performance issues directly from the artifacts.
- If the user already has profiler output or needs timing evidence for hotspot attribution, the skill should tell the agent to also use `skills/ascend-npu-operator-profiler/`.

## Archive layout

The capture script should create an IR directory with a stable, analysis-friendly layout:

- `triton_dump/`
  - copy of the extracted `dumped_ir_dir`
- `bishengir_stages/`
  - MLIR stage tree emitted by `--mlir-print-ir-tree-dir`
- `all-ir.txt`
  - full compiler stderr stream from the replayed compile command
- `capture-manifest.json`
  - machine-readable summary of:
    - the selected benchmark file
    - the selected operator file
    - the rendered benchmark command
    - whether the run was local or remote
    - the extracted `dumped_ir_dir`
    - the original `cmd_list`
    - the normalized replay command
    - the archived `kernel.ttadapter.mlir` path

If useful during implementation, the script may also persist the raw stdout or a rendered shell replay command, but those are secondary to the layout above.

## Local and remote execution

- Local mode should execute the rendered benchmark command in the benchmark file's workspace and archive artifacts locally.
- Remote mode should follow the same high-level semantics already used by the repository's remote operator-eval helpers:
  - accept `--remote user@host[:port]`
  - accept optional `--remote-workdir`
  - optionally keep the remote workspace through a dedicated keep flag
- Remote mode should run both the initial debug command and the Bisheng replay on the remote machine, then copy the IR directory back locally.
- If `--remote-workdir` is set, create a per-run subdirectory below that remote root instead of using a one-off temporary directory.
- Remote cleanup should remove only the workspace created by the current run, unless the keep flag is set.
- The local IR directory path should still be the canonical artifact location presented back to the agent and user.
- Remote mode should copy the selected benchmark harness and operator file into the remote workspace before execution.

## Design notes

- Keep the skill thin and procedural. The bundled script is the deterministic core for IR capture; the skill text should focus on when to use it, how to invoke it, and when to combine it with profiler evidence.
- Do not hard-code an IR performance methodology yet. The agent should perform free-form analysis from the archived IR because there is not yet a validated pattern library for Ascend-specific IR diagnosis.
- Reuse repository conventions where possible instead of inventing a new remote protocol:
  - shared SSH and copy behavior should stay aligned with the operator-eval runtime
  - command-line options should mirror existing remote flags when practical
- Keep the script resilient to shell quoting issues by parsing the extracted `cmd_list` into structured arguments before rewriting it.
- Treat missing `Dumping intermediate results to ...` or `[DEBUG] cmd_list: ...` lines as explicit failures with short actionable errors.

## Implementation outline

1. Create the new skill directory with a bundled `scripts/` folder.
2. Add a capture script that:
   - runs the rendered benchmark command with the required environment variables
   - extracts the dump path and compile command
   - copies dumped artifacts into the archive
   - rewrites and replays the compile command
   - supports local and remote execution
   - writes `capture-manifest.json`
3. Write `SKILL.md` so the agent:
   - runs the capture script first
   - analyzes archived IR directly
   - invokes the profiler skill when timing evidence is needed
4. Add lightweight tests for stdout parsing, command rewriting, manifest generation, and remote command construction.
5. Validate the new skill with the skill validator and the repository test/lint/typecheck commands.

## Verification

- Run the new script-focused unit tests.
- Run `python3 /Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py <skill-path>`.
- Run `uv run --group dev ruff check`.
- Run `uv run pyright`.
- Run `uv run python -m unittest discover -s tests -v`.
