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
uv run triton-agent run-test --test-file test_a.py --operator-file opt_a.py --test-mode differential
uv run triton-agent gen-bench --input a.py --bench-mode standalone
uv run triton-agent run-bench --bench-file bench_a.py --operator-file opt_a.py --bench-mode msprof
uv run triton-agent optimize --input a.py --test-mode differential --bench-mode standalone
uv run triton-agent gen-test --input a.py --verbose
uv run triton-agent gen-test --input a.py --show-output
uv run triton-agent gen-test --input a.py --force-overwrite
uv run triton-agent compare-result --oracle-result abs_result.pt --new-result opt_abs_result.pt
uv run triton-agent compare-perf --baseline abs_perf.txt --compare opt_abs_perf.txt
```

Generated harnesses record their resolved wrapper API, target kernel, and mode in a small file header such as `# test-mode: ...`, `# bench-mode: ...`, `# api-name: ...`, and `# kernel: ...`.
- Generated tests are expected to run directly as `python3 test_<op>.py --operator-file <path>` or `python3 differential_test_<op>.py --operator-file <path>`.
- Generated benchmarks are expected to run directly as `python3 bench_<op>.py --operator-file <path>` or, for msprof mode, `python3 bench_<op>.py --num-bench` and `python3 bench_<op>.py --operator-file <path> --bench <N>`.
- Generated harnesses should not require a runtime `--api-name` flag; the API comes from the generated file metadata.

- `--verbose` prints categorized diagnostics for files, skill links, and agent launch details.
- `--show-output` streams readable non-interactive agent output to the current terminal.
- `--show-output` exits cleanly after the agent finishes, including PTY-backed shutdown cases where Linux reports EOF as `EIO`.
- `--force-overwrite` makes the CLI delete an existing generated output file before starting `gen-test` or `gen-bench`.
- The parser also accepts snake_case command aliases such as `gen_test` and `run_bench`, while help text keeps the canonical kebab-case names.
- `run-test` requires `--test-file` and `--operator-file`.
- `run-test` executes the generated test file locally instead of launching a code agent.
- `run-test` streams local process output by default and does not support `--interact`.
- `run-test` prints the local process stdout, stderr, and return code.
- In `differential` mode, `run-test` archives the generated result payload beside the input operator file as `<operator-filename>_result.pt`.
- `run-bench` requires `--bench-file` and `--operator-file`.
- `run-bench` executes the generated benchmark file locally instead of launching a code agent.
- `run-bench` streams local process output by default and does not support `--interact`.
- In `standalone` mode, `run-bench` saves parsed `latency-<id>:` lines beside the input operator file as `<operator-filename>_perf.txt`.
- In `msprof` mode, `run-bench` first queries `--num-bench`, then executes one `msprof op --kernel-name=...` command per case using the benchmark metadata header.
- `compare-result` compares two archived differential result payload files directly.
- `compare-perf` compares two perf data files by `latency-<id>` and reports per-case deltas without relying on line order.
- The repository no longer keeps separate `test-run` or `bench-run` skills; `run-test` and `run-bench` are implemented directly in the CLI.
- `--test-mode` defaults to `standalone` for `gen-test` and `run-test`.
- `--bench-mode` defaults to `standalone` for `gen-bench` and `run-bench`.
- For `optimize`, `--test-mode` defaults to `differential` and `--bench-mode` defaults to `standalone`.
- Skill linking is idempotent: existing symlinks that already point to this repository's `skills/` tree are reused and left untouched.
- Codex non-interactive launches always include `--ephemeral` and `--skip-git-repo-check`.
- Codex uses `danger-full-access` for all non-interactive commands.
- The `optimize` workflow is expected to keep per-round artifacts under `opt-round-N/` and a top-level `opt-note.md` in the operator workspace.
- During `optimize`, the CLI writes a temporary workspace `AGENTS.md` with optimization guardrails; if the workspace already has one, it is backed up and restored after the run.
- The `optimize` skill is expected to choose optimization patterns through a compact pattern index before reading detailed pattern references.
- Skills can invoke the current checkout through the bundled helper script at `skills/scripts/run-command.py` without relying on an installed console entrypoint.
- If an output file already exists and overwrite is not allowed, the CLI prints a short error and exits without a Python traceback.

## Checks

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```
