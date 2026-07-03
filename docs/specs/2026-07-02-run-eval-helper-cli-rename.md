# Run-Eval Helper CLI Rename

## User-Visible Semantics

The `ascend-npu-run-eval` skill exposes its bundled command helper as
`scripts/cli.py`.

Agents should run:

```bash
python3 <ascend-npu-run-eval-skill-path>/scripts/cli.py <subcommand> ...
```

The subcommands and arguments keep their existing behavior. This change only
renames the script entrypoint so the skill follows the same `cli.py` convention
used by other command-oriented skills.

## Implementation

- Move `skills/common/ascend-npu-run-eval/scripts/run-command.py` to
  `skills/common/ascend-npu-run-eval/scripts/cli.py`.
- Update runtime callers to load `operator_eval_script_path("cli")`.
- Update live skill guidance, hook permissions, and tests to use `cli.py`.
- Leave historical design and plan documents unchanged unless they describe
  current live behavior.
