# triton-agent

## Usage

```bash
uv run triton-agent gen-test --input a.py
uv run triton-agent run-test --test-file test_a.py --operator-file a.py
uv run triton-agent gen-bench --input a.py
uv run triton-agent run-bench --bench-file bench_a.py --operator-file a.py
uv run triton-agent optimize --input a.py
uv run triton-agent optimize-status --input operators_root
uv run triton-agent optimize-batch --input operators_root
```

```bash
uv run triton-agent gen-test --input a.py --output test_a.py
uv run triton-agent optimize --input a.py --output opt_a.py --interact
uv run triton-agent gen-bench --input a.py --agent codex
uv run triton-agent gen-test --input a.py --agent opencode
uv run triton-agent gen-test --input a.py --agent pi
uv run triton-agent gen-test --input a.py --agent claude
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
uv run triton-agent optimize --input a.py --no-agent-session
uv run triton-agent optimize-status --input operators_root
uv run triton-agent optimize-batch --input operators_root --max-concurrency 4
uv run triton-agent optimize-batch --input operators_root --agent pi --test-mode differential --bench-mode standalone
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
- `run-test` executes the generated test file through the unified `skills/operator-eval/` execution helpers instead of launching a code agent.
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
- `run-bench` executes the generated benchmark file through the unified `skills/operator-eval/` execution helpers instead of launching a code agent.
- `run-bench` streams local process output by default and does not support `--interact`.
- `run-bench` no longer accepts `--agent`.
- `run-bench` reads `# bench-mode: ...` from the benchmark file metadata when `--bench-mode` is omitted.
- In `standalone` mode, `run-bench` saves parsed `latency-<id>:` lines beside the input operator file as `<operator-filename>_perf.txt`.
- In `msprof` mode, `run-bench` first queries `--num-bench`, then executes one `msprof op --kernel-name=...` command per case using the benchmark metadata header.
- The helper script `python3 skills/operator-eval/scripts/run-command.py profile-bench --bench-file <bench> --operator-file <operator>` profiles benchmark harnesses and prints a local `Profile directory: ...` plus the summarized operator report.
- `profile-bench` reads `# bench-mode: ...` from benchmark metadata when `--bench-mode` is omitted.
- In `standalone` mode, `profile-bench` wraps `python3 bench_<op>.py --operator-file <operator-file>` with `msprof` and must not receive `--bench`.
- In `msprof` mode, `profile-bench` first queries `--num-bench`, then profiles one selected `--bench <N>` case; this flow requires `# kernel: ...` metadata in the benchmark header.
- `profile-bench` also accepts `--remote user@host[:port]`, optional `--remote-workdir <path>`, and `--keep-remote-workdir`; remote runs copy the resulting `PROF_*` directory back beside the operator file before summarizing it locally.
- `compare-result` compares two archived differential result payload files through the unified `skills/operator-eval/` helpers.
- `compare-perf` compares two perf data files by `latency-<id>` and reports per-case deltas without relying on line order.
- The repository keeps a unified `run` skill for execution and comparison flows; the CLI stays thin and loads those skill-owned helpers dynamically.
- `--test-mode` defaults to `standalone` for `gen-test`; `run-test` infers it from test metadata unless you override it.
- `--bench-mode` defaults to `standalone` for `gen-bench`; `run-bench` infers it from benchmark metadata unless you override it.
- For `optimize`, `--test-mode` defaults to `differential` and `--bench-mode` defaults to `standalone`.
- `optimize` accepts `--min-rounds <N>` to require at least `N` `opt-round-*` directories before the run may finish successfully.
- `optimize` accepts `--continue` to resume an existing optimization session instead of starting a fresh one.
- `optimize` accepts `--no-agent-session` to request a non-persistent code-agent session when the selected backend supports it.
- `optimize-status` scans the immediate child directories under `--input` and treats each child directory as one operator workspace candidate.
- `optimize-status` is a local read-only summary command; it does not launch a code agent, support remote execution, or expose `--output` or `--interact`.
- `optimize-status` reports per-workspace numeric summaries including baseline mean latency, best mean latency, average improvement across per-case latency improvements, and both numeric-best and logged-best rounds when available.
- `optimize-status` keeps scanning when a workspace has missing or malformed optimize artifacts and reports those cases as warnings or no-session entries instead of aborting the whole batch.
- `optimize-batch` scans the immediate child directories under `--input` and treats each child directory as one operator workspace.
- In each batch workspace, `optimize-batch` auto-selects the only remaining `.py` file after excluding generated artifacts such as `test_*.py`, `differential_test_*.py`, `bench_*.py`, `opt_*.py`, and `__init__.py`.
- If a batch workspace has zero or multiple remaining `.py` candidates, `optimize-batch` reports that workspace as a failure and keeps processing the rest of the batch.
- `optimize-batch` accepts the same optimize orchestration flags as `optimize`, plus `--max-concurrency <N>` for bounded parallel execution.
- `optimize-batch` does not support `--output` or `--interact`.
- `optimize-batch --show-output` streams live per-workspace output into the current terminal with `[workspace-name] ` prefixes while still printing the compact batch summary at the end.
- For `optimize --no-agent-session`, Codex uses `--ephemeral`, Pi uses `--no-session`, and OpenCode ignores the flag.
- `optimize --continue` requires an existing `opt-note.md`, at least one `opt-round-*` directory, an existing generated test harness with readable `# test-mode: ...`, and an existing generated benchmark harness with readable `# bench-mode: ...`.
- `optimize --continue` rejects `--test-mode` and `--bench-mode`; it reuses the modes recorded in the existing harness metadata.
- If continue mode finds both `test_<op>.py` and `differential_test_<op>.py`, the CLI fails explicitly instead of guessing which harness should drive the resumed optimize run.
- If an `optimize` agent exits successfully before the workspace reaches the requested minimum round count, the supervisor automatically restarts the agent in continuation mode.
- `optimize-batch --continue` applies the same continue-mode validation to each workspace independently and summarizes workspace-level failures at the end.
- Continuation mode tells the agent to continue the existing optimization session and inspect `opt-note.md` plus existing `opt-round-*` artifacts before starting more work.
- Skill staging uses copied workspace content rather than symlinks, so code agents read ordinary workspace files without resolving back to the source repository.
- If a target workspace skill path already exists as a symlink, the CLI fails explicitly instead of reusing it.
- `workspace/` is a placeholder directory for local experimentation only; it is excluded from repository linting, static type checks, and test expectations.
- `uv run pyright` keeps repository-wide analysis enabled, applies strict checking to `src/`, and leaves `tests/` at the default basic level.
- Codex non-interactive launches always include `--skip-git-repo-check`.
- Codex non-interactive generation commands still include `--ephemeral`; `optimize` adds it only when `--no-agent-session` is requested.
- Codex uses `danger-full-access` for all non-interactive commands.
- Pi is available as an additional `--agent` backend on the existing agent-backed commands.
- Pi launches use `--thinking high` and `--no-extensions`.
- Pi generation commands still use `--no-session`; `optimize` adds it only when `--no-agent-session` is requested.
- Pi launches receive the staged workspace skill directory through `--skill .pi/skills` with `--no-skills` so repository-local skills stay authoritative.
- Claude is available as an additional `--agent` backend on the existing agent-backed commands.
- Claude discovers copied project skills from `.claude/skills`.
- Claude non-interactive launches use `--print --dangerously-skip-permissions`.
- For `optimize --no-agent-session`, Claude adds `--no-session-persistence` only in non-interactive mode; interactive Claude runs ignore that flag.
- The `optimize` workflow is expected to keep per-round artifacts under `opt-round-N/` and a top-level `opt-note.md` in the operator workspace.
- During `optimize`, the CLI writes a temporary workspace `AGENTS.md` with optimization guardrails; if the workspace already has one, it is backed up and restored after the run.
- The `optimize` skill is expected to choose optimization patterns through a compact pattern index before reading detailed pattern references.
- The `ascend-npu-operator-profiler` skill is expected to run profiling through direct `msprof <command>` invocation and summarize the generated `PROF_*/mindstudio_profiler_output/` CSV files instead of maintaining a separate benchmark-comparison analyzer.
- The `ascend-npu-operator-profiler` skill should prefer `python3 skills/operator-eval/scripts/run-command.py profile-bench ...` for generated benchmark harnesses, branch behavior by benchmark mode, and keep direct `msprof <command>` only as a manual fallback.
- Skills can invoke the current checkout through the bundled helper script at `skills/operator-eval/scripts/run-command.py` without relying on an installed console entrypoint.
- The scripts under `skills/operator-eval/scripts/` are standalone runtime modules and do not import `triton_agent`; the dependency direction is `triton_agent -> skills/operator-eval/scripts` only.
- If an output file already exists and overwrite is not allowed, the CLI prints a short error and exits without a Python traceback.

## Checks

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```

`uv run pyright` uses `typeCheckingMode = "basic"` for the repository by default and upgrades `src/` to strict checking through the `strict = ["src"]` path list, so contributor tests can stay less noisy while production code gets tighter enforcement.
