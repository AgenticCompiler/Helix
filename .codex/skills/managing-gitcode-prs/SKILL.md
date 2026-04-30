---
name: managing-gitcode-prs
description: Create, list, and inspect GitCode pull requests for the current project. Use when Codex needs to open a PR with `gc pr create`, review an existing PR with `gc pr view`, browse or filter PRs with `gc pr list`, or prepare PR metadata for a GitCode repository that authenticates through the `GC_TOKEN` environment variable.
---

# Managing GitCode PRs

Use this skill for GitCode pull request work only. Do not expand into general GitCode CLI guidance.

## Preconditions

Before running any `gc pr` command:

1. Confirm `GC_TOKEN` is present in the environment.
2. Confirm the target repository is known as `owner/repo`. For this workspace, prefer `midwinter1993/triton-agent` unless the user explicitly targets another repository.
3. For `pr create`, confirm the head branch is known. Prefer automatic current-branch detection, but fall back to `--head` when the current directory is not a Git repository or the branch cannot be resolved.

If `GC_TOKEN` is missing, stop and tell the user they need to set `GC_TOKEN="..."` before GitCode PR commands can authenticate.

If the repository is ambiguous, prefer an explicit `-R owner/repo` instead of guessing. In this repository, default to `-R midwinter1993/triton-agent` when the user does not name a different target.

Prefer the bundled wrapper at `<skill-path>/scripts/run-gc-pr.sh` over calling `gc pr` directly. The wrapper validates `GC_TOKEN`, sets a writable `UV_CACHE_DIR`, and runs `gc pr` through `uv tool run --from <wheel-url>`. It defaults to the provided GitCode CLI wheel URL and allows override through `GITCODE_CLI_WHEEL_URL`.

Read [references/pr-command-reference.md](./references/pr-command-reference.md) when you need exact flag shapes or example invocations.

## Default workflow

### Create a PR

- Prefer `gc pr create -R midwinter1993/triton-agent` for this workspace unless the user explicitly names another repo. Use explicit `--title` and `--body` when the user already knows the PR text.
- When invoking from this skill, prefer `bash <skill-path>/scripts/run-gc-pr.sh create ...`.
- Omit `--head` when the current Git branch is available and should be the PR head.
- Add `--head <branch>` when branch auto-detection is unavailable or when the user names a different head branch.
- Add `--base <branch>` only when the user wants a non-default base branch.
- Use `--fill` when the user wants to reuse the latest commit title/body and did not provide better PR text explicitly.
- Use `--draft` for work-in-progress PRs.
- Use `--fork <owner/repo>` only for explicit fork-to-upstream PR requests.
- Use `--web` only when the user explicitly wants the browser page opened after creation.

After creation, summarize the returned PR number, title, state, and URL for the user.

### List PRs

- Use `gc pr list -R midwinter1993/triton-agent` for this workspace unless the user explicitly names another repo.
- When invoking from this skill, prefer `bash <skill-path>/scripts/run-gc-pr.sh list ...`.
- Add filters such as `--state`, `--head`, `--base`, `--limit`, `--sort`, `--direction`, or `--page` when the request is specific.
- Prefer `--json` when the result will be parsed, summarized, or used in follow-up reasoning.
- Use `--format table` only when a human-readable table is more helpful than structured data.

### View a PR

- Use `gc pr view <number> -R midwinter1993/triton-agent` for this workspace unless the user explicitly names another repo.
- When invoking from this skill, prefer `bash <skill-path>/scripts/run-gc-pr.sh view ...`.
- Add `--comments` when the user wants discussion context as well as PR metadata.
- Prefer `--json` when extracting fields, comparing PRs, or feeding the result into another step.
- Use `--web` only when the user explicitly wants the PR opened in a browser.
- Use `--time-format relative` or `--time-format absolute` only for text-oriented display preferences; this does not change the JSON schema.

## Output rules

- Prefer structured output (`--json`) for machine-readable follow-up work.
- Prefer text output for short conversational summaries when no structured parsing is needed.
- Do not dump large raw command output into the conversation. Summarize the important fields instead.

## Failure handling

- If `GC_TOKEN` is unset, stop and report the missing environment variable clearly.
- If `gc pr create` cannot resolve the current branch, rerun with explicit `--head` rather than guessing.
- If the current directory is not a Git repository, require explicit `-R` and usually explicit `--head` for create flows.
- If the user asks to inspect a PR but does not provide a number and there is no unambiguous default, list matching PRs first instead of guessing.
- If a command fails because the repository slug is wrong, restate the expected `owner/repo` format and ask for the correct slug.

## Examples

Create a standard PR:

```bash
bash <skill-path>/scripts/run-gc-pr.sh create -R midwinter1993/triton-agent --title "New feature" --body "Description"
```

Create a draft PR from an explicit branch:

```bash
bash <skill-path>/scripts/run-gc-pr.sh create -R midwinter1993/triton-agent --head feature-branch --title "WIP: Feature" --draft
```

List merged PRs as JSON:

```bash
bash <skill-path>/scripts/run-gc-pr.sh list -R midwinter1993/triton-agent --state merged --json
```

View a PR with comments as JSON:

```bash
bash <skill-path>/scripts/run-gc-pr.sh view 1 -R midwinter1993/triton-agent --comments --json
```
