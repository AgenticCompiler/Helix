# triton-agent

## Usage

```bash
uv run triton-agent gen-test --input a.py
uv run triton-agent run-test --input a.py
uv run triton-agent gen-bench --input a.py
uv run triton-agent run-bench --input a.py
uv run triton-agent optimize --input a.py
```

```bash
uv run triton-agent gen-test --input a.py --output test_a.py
uv run triton-agent optimize --input a.py --output opt_a.py --interact
uv run triton-agent gen-bench --input a.py --agent codex
uv run triton-agent gen-test --input a.py --agent opencode
uv run triton-agent gen-test --input a.py --test-mode standalone
uv run triton-agent run-test --input a.py --test-mode differential
uv run triton-agent gen-bench --input a.py --bench-mode standalone
uv run triton-agent run-bench --input a.py --bench-mode msprof
uv run triton-agent optimize --input a.py --test-mode differential --bench-mode standalone
uv run triton-agent gen-test --input a.py --verbose
uv run triton-agent gen-test --input a.py --show-output
uv run triton-agent gen-test --input a.py --force-overwrite
```

`--verbose` prints categorized diagnostics for files, skill links, and agent launch details.
`--show-output` streams readable non-interactive agent output to the current terminal.
`--force-overwrite` makes the CLI delete an existing generated output file before starting `gen-test` or `gen-bench`.
`--test-mode` defaults to `standalone` for `gen-test` and `run-test`.
`--bench-mode` defaults to `standalone` for `gen-bench` and `run-bench`.
For `optimize`, `--test-mode` defaults to `differential` and `--bench-mode` defaults to `standalone`.
Skill linking is idempotent: existing symlinks that already point to this repository's `skills/` tree are reused and left untouched.
The `optimize` workflow is expected to keep per-round artifacts under `opt-round-N/` and a top-level `opt-note.md` in the operator workspace.
During `optimize`, the CLI writes a temporary workspace `AGENTS.md` with optimization guardrails; if the workspace already has one, it is backed up and restored after the run.
The `optimize` skill is expected to choose optimization patterns through a compact pattern index before reading detailed pattern references.
Skills can invoke the current checkout through the bundled helper script at `skills/scripts/run-command.py` without relying on an installed console entrypoint.
If an output file already exists and overwrite is not allowed, the CLI prints a short error and exits without a Python traceback.

## Checks

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m unittest discover -s tests -v
```
