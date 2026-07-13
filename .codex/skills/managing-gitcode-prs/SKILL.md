---
name: managing-gitcode-prs
description: Create, list, and inspect GitCode pull requests for the current project. Use when Codex needs to open a PR, review an existing PR, browse or filter PRs, or prepare PR metadata for a GitCode repository that authenticates through the `GC_TOKEN` environment variable and should use the official GitCode HTTP API.
---

# Managing GitCode PRs

Use this skill for GitCode pull request work only. Do not expand into general GitCode tooling guidance.

## Preconditions

Before running GitCode PR API requests:

1. Confirm `GC_TOKEN` is present in the environment.
2. Confirm the target repository is known as `owner/repo`. For this workspace, prefer `midwinter1993/triton-agent` unless the user explicitly targets another repository.
3. For PR creation, create a fresh topic branch for the change and use that branch as the PR head. Do not reuse an existing branch for a new PR.
4. For PR creation, confirm the head branch is known. Prefer current-branch auto-detection after switching to the fresh topic branch, but fall back to explicit `--head` when the current directory is not a Git repository or the branch cannot be resolved.

If `GC_TOKEN` is missing, stop and tell the user they need to set `GC_TOKEN="..."` before GitCode PR API requests can authenticate.

If the repository is ambiguous, prefer explicit `-R owner/repo` instead of guessing. In this repository, default to `-R midwinter1993/triton-agent` when the user does not name a different target.

Prefer the bundled script at `<skill-path>/scripts/gitcode_pr_api.py` over ad hoc HTTP snippets. The script:

- authenticates with `Authorization: Bearer $GC_TOKEN`
- targets `https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls`
- supports `create`, `list`, and `view`
- retries once without proxy settings when local proxy variables point at a dead `127.0.0.1` or `localhost` proxy

Read [references/pr-command-reference.md](./references/pr-command-reference.md) when you need exact flags or endpoint mappings.

## Default workflow

### Create a PR

- Prefer `python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent ...` for this workspace unless the user explicitly names another repo.
- Use explicit `--title` and `--body` when the user already knows the PR text.
- Create a fresh topic branch for the PR first, then use that branch as the PR head.
- Omit `--head` only when the current Git branch is the fresh topic branch and should become the PR head.
- Add `--head <branch>` when branch auto-detection is unavailable or when you want to make the fresh topic branch explicit.
- Add `--base <branch>` only when the user wants a non-default target branch.
- Use `--fill` when the user wants to reuse the latest commit title/body and did not provide better PR text explicitly.
- Use `--draft` when the user wants the created PR immediately marked as draft.
- Use `--prune-source-branch` when the user wants GitCode to delete the source branch after merge.
- Use `--json` when a downstream step needs the raw API response.

After creation, summarize the returned PR number, title, state, and URL for the user.

### List PRs

- Use `python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent ...` for this workspace unless the user explicitly names another repo.
- Add filters such as `--state`, `--head`, `--base`, `--limit`, `--page`, `--sort`, or `--direction` when the request is specific.
- Prefer `--json` when the result will be parsed, summarized, or used in follow-up reasoning.

### View a PR

- Use `python3 <skill-path>/scripts/gitcode_pr_api.py view <number> -R midwinter1993/triton-agent ...` for this workspace unless the user explicitly names another repo.
- Add `--comments` when the user wants discussion context as well as PR metadata.
- Prefer `--json` when extracting fields, comparing PRs, or feeding the result into another step.
- Use `--time-format relative` when a concise human summary is more useful than absolute timestamps.

## Output rules

- Prefer structured output (`--json`) for machine-readable follow-up work.
- Prefer text output for short conversational summaries when no structured parsing is needed.
- Do not dump large raw API responses into the conversation. Summarize the important fields instead.

## Failure handling

- If `GC_TOKEN` is unset, stop and report the missing environment variable clearly.
- If PR creation cannot resolve the current branch, rerun with explicit `--head` rather than guessing.
- If the current directory is not a Git repository, require explicit `-R` and usually explicit `--head` for create flows.
- If the user asks to inspect a PR but does not provide a number and there is no unambiguous default, list matching PRs first instead of guessing.
- If the request fails while local proxy environment variables point at a dead loopback proxy, retry once with those proxy settings disabled.
- If a request fails because the repository slug is wrong, restate the expected `owner/repo` format and ask for the correct slug.

## Examples

Create a standard PR:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --title "New feature" --body "Description"
```

Create a draft PR from an explicit branch:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py create -R midwinter1993/triton-agent --head feature-branch --title "WIP: Feature" --draft
```

List merged PRs as JSON:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py list -R midwinter1993/triton-agent --state merged --json
```

View a PR with comments as JSON:

```bash
python3 <skill-path>/scripts/gitcode_pr_api.py view 1 -R midwinter1993/triton-agent --comments --json
```
