# `run-test`

Run a generated test with:

```bash
python3 ./scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py
python3 ./scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py
```

Rules:

- Always pass both `--test-file` and `--operator-file`.
- If `--test-mode` is omitted, the command reads `# test-mode: ...` from the test file.
- Use `--test-mode standalone` or `--test-mode differential` only when you need to override the embedded metadata.
- In `differential` mode, generated tests are import-only modules. `run-test` imports the module, calls `build_operator_api(operator_module)`, then calls `build_differential_test_cases(operator_api)` and archives the result as `<operator>_result.pt`.
- Existing legacy script-style differential tests are still supported for compatibility, but newly generated differential tests should use the import-only hook contract.
- When a differential run succeeds, the command prints the archived result path and a hint to use `compare-result`.

Examples:

```bash
python3 ./scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py --test-mode standalone
python3 ./scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --test-mode differential
```

Remote examples:

```bash
python3 ./scripts/run-command.py run-test --test-file test_<operator>.py --operator-file <operator>.py --remote user@host:2222
python3 ./scripts/run-command.py run-test --test-file differential_test_<operator>.py --operator-file opt_<operator>.py --remote user@host:2222 --remote-workdir /tmp/triton-agent
```
