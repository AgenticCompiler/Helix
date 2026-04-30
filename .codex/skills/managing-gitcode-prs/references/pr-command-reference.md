# GitCode PR API Reference

Use this reference only when you need the exact script flags or official endpoint mapping. Keep `SKILL.md` as the workflow entrypoint.

For this workspace, prefer `-R midwinter1993/triton-agent` unless the user explicitly targets another repository.

## Preferred launcher for this skill

Prefer the bundled script:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py <create|list|view> ...
```

It authenticates with:

```text
Authorization: Bearer $GC_TOKEN
```

And targets:

```text
https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls
```

## Authentication

- Required environment variable: `GC_TOKEN`
- Example:

```bash
export GC_TOKEN="your_gitcode_token"
```

## Official endpoints

- List PRs:

```text
GET /api/v5/repos/{owner}/{repo}/pulls
```

- Create PR:

```text
POST /api/v5/repos/{owner}/{repo}/pulls
```

- View one PR:

```text
GET /api/v5/repos/{owner}/{repo}/pulls/{number}
```

- View PR comments:

```text
GET /api/v5/repos/{owner}/{repo}/pulls/{number}/comments
```

- Update PR metadata such as draft state:

```text
PATCH /api/v5/repos/{owner}/{repo}/pulls/{number}
```

## `create`

Standard create:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --title "New feature" --body "Description"
```

Explicit head:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --head feature-branch --title "Feature" --body "Description"
```

Explicit base:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --base main --title "Feature" --body "Description"
```

Draft:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --title "WIP: Feature" --draft
```

Fill from latest commit:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --fill
```

Structured JSON output:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --title "New feature" --body "Description" --json
```

Notes:

- `--head` is optional when the current branch can be detected from the current Git repository.
- `--fill` uses the latest Git commit title and body to supply missing PR text.
- `--draft` creates the PR first, then patches it into draft state through the official API.
- Current branch resolution depends on being in a valid Git repo with a recognizable branch. When that fails, use explicit `--head`.

## `list`

Open PRs:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent
```

Closed or merged:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent --state closed
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent --state merged
```

Filter by branches:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent --head feature/login --base main
```

Limit, sorting, pagination:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent --limit 10
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent --sort updated --direction desc --page 2
```

Structured output:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent --json
```

## `view`

Basic details:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py view 1 -R midwinter1993/triton-agent
```

With comments:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py view 1 -R midwinter1993/triton-agent --comments
```

JSON output:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py view 1 -R midwinter1993/triton-agent --json
python3 <skill-path>/scripts/gitcode_pr_api.py view 1 -R midwinter1993/triton-agent --comments --json
```

Time formatting for text output:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py view 1 -R midwinter1993/triton-agent --time-format relative
python3 <skill-path>/scripts/gitcode_pr_api.py view 1 -R midwinter1993/triton-agent --time-format absolute
```

Notes:

- `--comments` issues an additional request to the official PR comments endpoint.
- `--time-format` changes text rendering only; it does not change JSON structure.
- Prefer `--json` when the next step needs reliable field extraction.
