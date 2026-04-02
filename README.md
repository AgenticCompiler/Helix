# triton-agent

## Usage

```bash
uv run triton-agent gen-test --input a.py
uv run triton-agent run-test --test-file test_a.py --operator-file a.py
uv run triton-agent gen-bench --input a.py
uv run triton-agent run-bench --bench-file bench_a.py --operator-file a.py
uv run triton-agent optimize --input a.py
```

```bash
uv run triton-agent gen-test --input a.py --output test_a.py
uv run triton-agent optimize --input a.py --output opt_a.py --interact
uv run triton-agent gen-bench --input a.py --agent codex
uv run triton-agent gen-test --input a.py --agent opencode
uv run triton-agent gen-test --input a.py --test-mode standalone
uv run triton-agent run-test --test-file differential_test_a.py --operator-file opt_a.py
uv run triton-agent gen-bench --input a.py --bench-mode standalone
uv run triton-agent run-bench --bench-file bench_a.py --operator-file opt_a.py
uv run triton-agent optimize --input a.py --test-mode differential --bench-mode standalone
uv run triton-agent gen-test --input a.py --verbose
uv run triton-agent gen-test --input a.py --show-output
uv run triton-agent gen-test --input a.py --force-overwrite
uv run triton-agent compare-result --oracle-result abs_result.pt --new-result opt_abs_result.pt
uv run triton-agent compare-perf --baseline abs_perf.txt --compare opt_abs_perf.txt
uv run triton-agent run-test --test-file test_a.py --operator-file a.py --remote user@host:2222
uv run triton-agent run-bench --bench-file bench_a.py --operator-file a.py --remote user@host
uv run triton-agent compare-result --oracle-result abs_result.pt --new-result opt_abs_result.pt --remote user@host --remote-workdir /tmp/triton-agent
uv run triton-agent gen-test --input a.py --remote user@host:2222
uv run triton-agent gen-bench --input a.py --remote user@host --remote-workdir /tmp/triton-agent
uv run triton-agent optimize --input a.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
uv run triton-agent optimize --input a.py --min-rounds 3
uv run triton-agent optimize --input a.py --continue
```

Generated harnesses record their resolved public entrypoint, entrypoint kind, target kernel, and mode in a small file header such as `# test-mode: ...`, `# bench-mode: ...`, `# api-name: ...`, `# api-kind: ...`, and `# kernel: ...`.
- Generated tests are expected to run directly as `python3 test_<op>.py --operator-file <path>` or `python3 differential_test_<op>.py --operator-file <path>`.
- Generated benchmarks are expected to run directly as `python3 bench_<op>.py --operator-file <path>` or, for msprof mode, `python3 bench_<op>.py --num-bench` and `python3 bench_<op>.py --operator-file <path> --bench <N>`.
- Generated harnesses should not require a runtime `--api-name` flag; the runtime entrypoint comes from the generated file metadata.
- `# api-kind:` distinguishes `triton-wrapper`, `torch-function`, and `torch-module` entrypoints so the harness can load or instantiate the target correctly.
- `torch-module` entrypoints currently require no-argument construction; the generators should fail explicitly instead of guessing constructor arguments.

- `--verbose` prints categorized diagnostics for files, skill staging, and agent launch details.
- `--show-output` streams readable non-interactive agent output to the current terminal.
- `--show-output` exits cleanly after the agent finishes, including PTY-backed shutdown cases where Linux reports EOF as `EIO`.
- `--force-overwrite` makes the CLI delete an existing generated output file before starting `gen-test` or `gen-bench`.
- The parser also accepts snake_case command aliases such as `gen_test` and `run_bench`, while help text keeps the canonical kebab-case names.
- `run-test` requires `--test-file` and `--operator-file`.
- `run-test` executes the generated test file through the unified `skills/run-validation/` execution helpers instead of launching a code agent.
- `run-test` streams local process output by default and does not support `--interact`.
- `run-test` no longer accepts `--agent`.
- `run-test` prints the local process stdout, stderr, and return code.
- `run-test` reads `# test-mode: ...` from the test file metadata when `--test-mode` is omitted.
- In `differential` mode, `run-test` archives the generated result payload beside the input operator file as `<operator-filename>_result.pt`.
- `run-test`, `run-bench`, and `compare-result` accept `--remote user@host[:port]` to execute through SSH on a remote machine.
- `--remote-workdir <path>` makes each remote run create a subdirectory under the given remote root; otherwise the CLI uses a temporary remote directory.
- `run-test` and `run-bench` accept `--keep-remote-workdir` to skip remote cleanup and print the retained remote workspace path for debugging.
- `gen-test`, `gen-bench`, and `optimize` also accept `--remote user@host[:port]` and optional `--remote-workdir <path>`.
- For `gen-test`, `gen-bench`, and `optimize`, the code agent still runs locally; the remote settings are passed through prompt context so the skills use remote-aware repository commands during validation.
- Under `--verbose`, remote runs print the concrete `ssh` and `scp` commands they execute.
- `run-bench` requires `--bench-file` and `--operator-file`.
- `run-bench` executes the generated benchmark file through the unified `skills/run-validation/` execution helpers instead of launching a code agent.
- `run-bench` streams local process output by default and does not support `--interact`.
- `run-bench` no longer accepts `--agent`.
- `run-bench` reads `# bench-mode: ...` from the benchmark file metadata when `--bench-mode` is omitted.
- In `standalone` mode, `run-bench` saves parsed `latency-<id>:` lines beside the input operator file as `<operator-filename>_perf.txt`.
- In `msprof` mode, `run-bench` first queries `--num-bench`, then executes one `msprof op --kernel-name=...` command per case using the benchmark metadata header.
- `compare-result` compares two archived differential result payload files through the unified `skills/run-validation/` helpers.
- `compare-perf` compares two perf data files by `latency-<id>` and reports per-case deltas without relying on line order.
- The repository keeps a unified `run` skill for execution and comparison flows; the CLI stays thin and loads those skill-owned helpers dynamically.
- `--test-mode` defaults to `standalone` for `gen-test`; `run-test` infers it from test metadata unless you override it.
- `--bench-mode` defaults to `standalone` for `gen-bench`; `run-bench` infers it from benchmark metadata unless you override it.
- For `optimize`, `--test-mode` defaults to `differential` and `--bench-mode` defaults to `standalone`.
- `optimize` accepts `--min-rounds <N>` to require at least `N` `opt-round-*` directories before the run may finish successfully.
- `optimize` accepts `--continue` to resume an existing optimization session instead of starting a fresh one.
- `optimize --continue` requires an existing `opt-note.md`, at least one `opt-round-*` directory, an existing generated test harness with readable `# test-mode: ...`, and an existing generated benchmark harness with readable `# bench-mode: ...`.
- `optimize --continue` rejects `--test-mode` and `--bench-mode`; it reuses the modes recorded in the existing harness metadata.
- If continue mode finds both `test_<op>.py` and `differential_test_<op>.py`, the CLI fails explicitly instead of guessing which harness should drive the resumed optimize run.
- If an `optimize` agent exits successfully before the workspace reaches the requested minimum round count, the supervisor automatically restarts the agent in continuation mode.
- Continuation mode tells the agent to continue the existing optimization session and inspect `opt-note.md` plus existing `opt-round-*` artifacts before starting more work.
- Skill staging uses copied workspace content rather than symlinks, so code agents read ordinary workspace files without resolving back to the source repository.
- If a target workspace skill path already exists as a symlink, the CLI fails explicitly instead of reusing it.
- `workspace/` is a placeholder directory for local experimentation only; it is excluded from repository linting, static type checks, and test expectations.
- Codex non-interactive launches always include `--ephemeral` and `--skip-git-repo-check`.
- Codex uses `danger-full-access` for all non-interactive commands.
- The `optimize` workflow is expected to keep per-round artifacts under `opt-round-N/` and a top-level `opt-note.md` in the operator workspace.
- During `optimize`, the CLI writes a temporary workspace `AGENTS.md` with optimization guardrails; if the workspace already has one, it is backed up and restored after the run.
- The `optimize` skill is expected to choose optimization patterns through a compact pattern index before reading detailed pattern references.
- The `ascend-npu-operator-profiler` skill is expected to run profiling through direct `msprof <command>` invocation and summarize the generated `PROF_*/mindstudio_profiler_output/` CSV files instead of maintaining a separate benchmark-comparison analyzer.
- Skills can invoke the current checkout through the bundled helper script at `skills/run-validation/scripts/run-command.py` without relying on an installed console entrypoint.
- The scripts under `skills/run-validation/scripts/` are standalone runtime modules and do not import `triton_agent`; the dependency direction is `triton_agent -> skills/run-validation/scripts` only.
- If an output file already exists and overwrite is not allowed, the CLI prints a short error and exits without a Python traceback.

## Checks

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```
