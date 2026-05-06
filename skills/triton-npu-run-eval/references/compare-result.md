# `compare-result`

If the test mode is `differential`, compare the archived result payloads after `run-test` succeeds:

```bash
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt>
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt> --compare-level balanced
```

Rules:

- Use this flow only after you already have the archived oracle and candidate result payloads.
- `--compare-level balanced` is an optional stricter comparison setting when you do not want the default level.

Remote example:

```bash
python3 ./scripts/run-command.py compare-result --oracle-result <oracle_result.pt> --new-result <new_result.pt> --remote user@host:2222 --remote-workdir /tmp/triton-agent
```
