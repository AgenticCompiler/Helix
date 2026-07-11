# GitCode PR Official API Design

## Summary

Replace the repo-local `gitcode-cli` wrapper approach with a repo-local official API script for the `managing-gitcode-prs` skill.

## Motivation

The skill no longer needs GitCode CLI now that the official PR API shape is known and reachable from this environment. The CLI path introduced two avoidable failure modes: a private wheel URL that returned `401 Unauthorized`, and a packaged `gc` launcher whose platform binary lookup failed on macOS arm64. The official API succeeded with header-based authentication, so the skill should prefer that stable path.

## Decision

- Remove the `run-gc-pr.sh` wrapper and all `uv tool run` / wheel guidance.
- Add a Python script under `.codex/skills/managing-gitcode-prs/scripts/` that uses the official GitCode PR API.
- Authenticate with `Authorization: Bearer $GC_TOKEN` instead of embedding tokens in URLs.
- Keep the skill focused on create, list, and view PR workflows.
- Preserve the workspace default repository of `midwinter1993/helix`.

## API Surface

- List PRs: `GET https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls`
- Create PR: `POST https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls`
- Get single PR: `GET https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}`
- Get PR comments: `GET https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments`
- Update PR draft state when needed: `PATCH https://gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}`

## Script Shape

- Add `scripts/gitcode_pr_api.py`.
- Support subcommands `create`, `list`, and `view`.
- Default `-R/--repo` to `midwinter1993/helix`.
- For `create`:
  - auto-detect the current branch when `--head` is omitted
  - support `--fill` from the latest git commit title/body
  - support `--draft` by following creation with a patch request
- For `view`:
  - support optional `--comments`
  - support `--json` output for structured consumers

## Failure Handling

- Require `GC_TOKEN`.
- If branch auto-detection fails, require explicit `--head`.
- If network access fails while proxy variables point to a dead local proxy such as `127.0.0.1`, retry once without proxy settings.
- Report non-2xx API responses with actionable error text.

## Verification

- Replace the contract test so it requires the official API script and bans the old wrapper.
- Run the targeted contract test.
- Run `bash scripts/run-skill-script-pyright.sh .codex/skills/managing-gitcode-prs/scripts/gitcode_pr_api.py`.
- Re-run the skill validator.
