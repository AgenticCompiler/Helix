# Heredoc Guard And Claude Session Persistence

## Context

Two runtime behaviors are currently surprising:

1. Shell guard logic treats `cat > file <<EOF ... EOF` as a read-oriented command and scans the here-doc body for protected paths, which can wrongly block ordinary writes such as `learned_lessons.md`.
2. Claude optimize invocations appear to default to `--no-session-persistence` even when the user did not request `--no-agent-session`.

## Design

### Shell Guard Read Scanning

Keep the existing read-deny policy, but narrow the shell text that is scanned for read paths:

- skip output redirection targets such as `> out.txt` and `2> err.txt`
- skip here-doc payload text after `<<EOF` / `<<'EOF'`
- continue checking true read inputs such as `cat outside.txt > out.txt` and `cat < outside.txt`

Apply the same behavior to both shared Python guard logic and the Opencode JavaScript hook so backend behavior stays aligned.

### Optimize Session Persistence

Stop forcing `no_agent_session=True` inside optimize execution. Worker and supervisor invocations should inherit the request-level `no_agent_session` choice that came from CLI or caller configuration.

This preserves explicit opt-out behavior while removing the current hidden default.

## Verification

- Add regression coverage showing here-doc writes remain allowed even when the body mentions protected runtime paths.
- Add regression coverage showing redirected reads from outside the workspace are still denied.
- Add optimize runtime coverage showing worker and supervisor requests preserve `no_agent_session=False` by default.
- Run focused pytest coverage, plus repository lint/type checks and the required strict pyright check for modified skill scripts when applicable.
