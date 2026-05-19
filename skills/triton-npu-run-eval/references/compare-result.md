# `compare-result`

Use this command when you already have both archived payloads and want to rerun or inspect the comparison separately from `run-test`:

```bash
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt>
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt> --compare-level balanced
```

Rules:

- Use this flow only after you already have the archived oracle and candidate result payloads.
- Prefer `run-test --oracle-result ...` when you want the agent to execute the differential run and the result comparison in one command.
- `--compare-level balanced` is an optional stricter comparison setting when you do not want the default level.

Remote example:

```bash
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt> --remote user@host:2222 --remote-workdir /tmp/triton-agent
```
