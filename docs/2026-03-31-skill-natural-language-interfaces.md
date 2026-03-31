## Summary

- Rewrite the operator workflow skills so they describe natural-language task inputs first instead of presenting CLI flags as their primary interface.
- Keep CLI flag names only as optional outer-wrapper context when they help map external orchestration onto the skill.

## User-Visible Semantics

- Skills should read like instructions for another agent receiving a natural-language request.
- Paths, modes, output destinations, and overwrite intent are still valid inputs, but they should be described as information the user or wrapper provides rather than as command-line switches.

## Implementation Notes

- Update frontmatter descriptions to remove flag-oriented trigger wording.
- Rewrite `Inputs`, `Option Contract`, and `Workflow` sections to refer to requested paths, execution modes, and destinations in plain language.
- Keep mode names such as `standalone`, `differential`, and `msprof`, because those are stable task concepts rather than CLI-only syntax.
