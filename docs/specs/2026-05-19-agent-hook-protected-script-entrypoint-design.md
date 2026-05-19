# Agent Hook Protected Script Entrypoint Design

## Goal

Keep the opt-in agent hook guard focused on blocking source inspection, while
still allowing the documented staged helper script entrypoints to execute.

## User-Visible Behavior

- `optimize --enable-agent-hooks` should continue blocking direct reads of
  staged skill implementation scripts under `.codex/skills/*/scripts/**` and
  `.opencode/skills/*/scripts/**`.
- The same guarded runs should allow commands that execute one of those staged
  scripts as the Python program entrypoint, such as
  `python3 .opencode/skills/.../scripts/run-command.py ...`.
- The denial message stays unchanged.

## Design

- Treat direct file reads and inline source inspection as protected reads.
  Examples: `cat`, `sed`, direct `Read`, and Python one-liners that call
  `open(...).read()`.
- Add a narrow exception for `python` and `python3` commands whose first script
  path argument resolves to an in-workspace protected staged script. That token
  represents the documented command interface entrypoint, not a source-inspect
  target.
- Keep evaluating every other candidate path in the same command, so outside
  workspace reads and protected-path reads hidden in other arguments still get
  denied.

## Scope Boundaries

- Do not broaden access to other staged skill files.
- Do not treat arbitrary interpreter usage as safe.
- Do not change skill staging, runner command construction, or denial wording.
