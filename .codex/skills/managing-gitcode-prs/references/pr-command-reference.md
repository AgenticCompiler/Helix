# GitCode PR Command Reference

Use this reference only when you need the exact `gc pr` flag combinations. Keep `SKILL.md` as the workflow entrypoint.

For this workspace, prefer `-R midwinter1993/triton-agent` unless the user explicitly targets another repository.

## Preferred launcher for this skill

Prefer the bundled wrapper:

```bash
bash <skill-path>/scripts/run-gc-pr.sh <create|list|view> ...
```

It validates `GC_TOKEN`, sets a writable `UV_CACHE_DIR`, and runs `gc pr` through:

```bash
uv tool run --from "$GITCODE_CLI_WHEEL_URL" gc pr ...
```

The default `GITCODE_CLI_WHEEL_URL` points at the provided `gitcode_cli-0.3.11` wheel URL, and you may override it through the environment when needed.

## Authentication

- Required environment variable: `GC_TOKEN`
- Example:

```bash
export GC_TOKEN="your_gitcode_token"
```

## `gc pr create`

Standard create:

```bash
gc pr create -R midwinter1993/triton-agent --title "New feature" --body "Description"
```

Explicit head:

```bash
gc pr create -R midwinter1993/triton-agent --head feature-branch --title "Feature" --body "Description"
```

Explicit base:

```bash
gc pr create -R midwinter1993/triton-agent --base main --title "Feature" --body "Description"
```

Draft:

```bash
gc pr create -R midwinter1993/triton-agent --title "WIP: Feature" --draft
```

Cross-repo from fork:

```bash
gc pr create -R upstream/repo --fork myfork/repo --head feature-branch --title "Feature"
```

Fill from latest commit:

```bash
gc pr create -R midwinter1993/triton-agent --fill
```

Open in browser after creation:

```bash
gc pr create -R midwinter1993/triton-agent --title "New feature" --body "Description" --web
```

Notes:

- `--head` is optional when the current branch can be detected from the current Git repository.
- `--fill` uses the latest Git commit title and body to supply missing PR text.
- `--web` opens the created PR page in a browser.
- Current branch resolution depends on being in a valid Git repo with a recognizable branch. When that fails, use explicit `--head`.

## `gc pr list`

Open PRs:

```bash
gc pr list -R midwinter1993/triton-agent
```

Closed or merged:

```bash
gc pr list -R midwinter1993/triton-agent --state closed
gc pr list -R midwinter1993/triton-agent --state merged
```

Filter by branches:

```bash
gc pr list -R midwinter1993/triton-agent --head feature/login --base main
```

Limit, sorting, pagination:

```bash
gc pr list -R midwinter1993/triton-agent --limit 10
gc pr list -R midwinter1993/triton-agent --sort updated --direction desc --page 2
```

Structured or table output:

```bash
gc pr list -R midwinter1993/triton-agent --json
gc pr list -R midwinter1993/triton-agent --format table
```

## `gc pr view`

Basic details:

```bash
gc pr view 1 -R midwinter1993/triton-agent
```

With comments:

```bash
gc pr view 1 -R midwinter1993/triton-agent --comments
```

Open in browser:

```bash
gc pr view 1 -R midwinter1993/triton-agent --web
```

JSON output:

```bash
gc pr view 1 -R midwinter1993/triton-agent --json
gc pr view 1 -R midwinter1993/triton-agent --comments --json
```

Time formatting for text output:

```bash
gc pr view 1 -R midwinter1993/triton-agent --time-format relative
gc pr view 1 -R midwinter1993/triton-agent --time-format absolute
```

Notes:

- The text detail layout is intended to be stable for human and agent reading.
- `--time-format` changes text rendering only; it does not change JSON structure.
- Prefer `--json` when the next step needs reliable field extraction.
