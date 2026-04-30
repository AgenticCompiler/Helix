# GitCode PR UV Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local wrapper that runs GitCode PR commands through `uv` for the `managing-gitcode-prs` skill, without adding GitCode CLI to project dev dependencies.

**Architecture:** Keep the behavior inside the repo-local skill. Add one shell script under `.codex/skills/managing-gitcode-prs/scripts/`, point `SKILL.md` and the command reference at that script, and protect the contract with a focused unittest in `tests/test_generation_contracts.py`.

**Tech Stack:** Bash, `uv`, Markdown docs, Python `unittest`

---

### Task 1: Record the wrapper contract

**Files:**
- Create: `docs/specs/2026-04-30-gitcode-pr-wrapper-design.md`
- Create: `docs/plans/2026-04-30-gitcode-pr-wrapper.md`

- [ ] **Step 1: Write the short design document**

Document why the wrapper belongs in the skill, why GitCode CLI should stay out of `[dependency-groups].dev`, and which environment variables and `uv` command shape the wrapper owns.

- [ ] **Step 2: Write the implementation plan**

Record the test-first flow, the wrapper file path, and the verification commands.

### Task 2: Lock the contract with a failing test

**Files:**
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Add a failing contract test for the GitCode PR wrapper**

Assert that:
- `.codex/skills/managing-gitcode-prs/scripts/run-gc-pr.sh` exists
- `SKILL.md` mentions the wrapper
- the wrapper contains `uv tool run --from`, `gc pr`, and `GC_TOKEN`

- [ ] **Step 2: Run the targeted test and confirm it fails**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_gitcode_pr_skill_uses_repo_local_uv_wrapper -v
```

Expected: FAIL because the wrapper script and related doc text do not exist yet.

### Task 3: Implement the wrapper and doc updates

**Files:**
- Create: `.codex/skills/managing-gitcode-prs/scripts/run-gc-pr.sh`
- Modify: `.codex/skills/managing-gitcode-prs/SKILL.md`
- Modify: `.codex/skills/managing-gitcode-prs/references/pr-command-reference.md`

- [ ] **Step 1: Add the wrapper script**

Make it:
- validate `GC_TOKEN`
- set `UV_CACHE_DIR` if missing
- default `GITCODE_CLI_WHEEL_URL` to the provided v0.3.11 wheel URL
- exec `uv tool run --from "$GITCODE_CLI_WHEEL_URL" gc pr "$@"`

- [ ] **Step 2: Update `SKILL.md`**

Tell agents to prefer `./scripts/run-gc-pr.sh ...` over raw `gc pr ...`, while keeping the command semantics unchanged.

- [ ] **Step 3: Update the command reference**

Show wrapper-based examples alongside the underlying `gc pr` equivalents.

### Task 4: Verify the result

**Files:**
- Modify: `tests/test_generation_contracts.py` as needed

- [ ] **Step 1: Run the targeted contract test**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_gitcode_pr_skill_uses_repo_local_uv_wrapper -v
```

Expected: PASS.

- [ ] **Step 2: Validate shell syntax**

Run:

```bash
bash -n .codex/skills/managing-gitcode-prs/scripts/run-gc-pr.sh
```

Expected: exit 0.

- [ ] **Step 3: Re-run the skill validator**

Run:

```bash
uv run python /Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py /Users/cdj/Projects/triton-agent/.codex/skills/managing-gitcode-prs
```

Expected: `Skill is valid!`
