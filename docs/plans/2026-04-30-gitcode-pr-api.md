# GitCode PR Official API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the `managing-gitcode-prs` skill from the GitCode CLI wrapper to a repo-local script that uses the official GitCode PR API.

**Architecture:** Keep all behavior inside the repo-local skill. Add one Python script under `.codex/skills/managing-gitcode-prs/scripts/`, point `SKILL.md` and the command reference at that script, and guard the contract with one focused unittest.

**Tech Stack:** Python stdlib, Markdown docs, Python `unittest`

---

### Task 1: Record the API contract

**Files:**
- Create: `docs/specs/2026-04-30-gitcode-pr-api-design.md`
- Create: `docs/plans/2026-04-30-gitcode-pr-api.md`
- Delete: `docs/specs/2026-04-30-gitcode-pr-wrapper-design.md`
- Delete: `docs/plans/2026-04-30-gitcode-pr-wrapper.md`

- [ ] **Step 1: Replace the obsolete wrapper docs**

Document the move from `gitcode-cli` to the official PR API and remove the obsolete wrapper-specific design and plan files.

### Task 2: Lock the new contract with a failing test

**Files:**
- Modify: `tests/test_generation_contracts.py`

- [ ] **Step 1: Replace the wrapper contract test**

Require:
- `.codex/skills/managing-gitcode-prs/scripts/gitcode_pr_api.py`
- `SKILL.md` and the reference file to mention the API script
- the script to reference `Authorization`, `Bearer`, `GC_TOKEN`, and `https://gitcode.com/api/v5/repos`
- the old `run-gc-pr.sh` wrapper text to be absent

- [ ] **Step 2: Run the targeted test and confirm it fails**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_gitcode_pr_skill_uses_official_api_script -v
```

Expected: FAIL because the API script and updated docs do not exist yet.

### Task 3: Implement the API script and doc updates

**Files:**
- Delete: `.codex/skills/managing-gitcode-prs/scripts/run-gc-pr.sh`
- Create: `.codex/skills/managing-gitcode-prs/scripts/gitcode_pr_api.py`
- Modify: `.codex/skills/managing-gitcode-prs/SKILL.md`
- Modify: `.codex/skills/managing-gitcode-prs/references/pr-command-reference.md`
- Modify: `docs/specs/2026-04-29-gitcode-pr-skill-design.md`

- [ ] **Step 1: Add the API script**

Implement `create`, `list`, and `view` on top of the official GitCode PR API with header-based auth and a default repo of `midwinter1993/helix`.

- [ ] **Step 2: Update the skill docs**

Remove `gc pr` and `uv tool run` guidance. Replace it with API-script usage and official endpoint guidance.

- [ ] **Step 3: Update the base design doc**

Make the stable skill design describe the official API path instead of GitCode CLI.

### Task 4: Verify the result

**Files:**
- Modify: `tests/test_generation_contracts.py` as needed

- [ ] **Step 1: Run the targeted contract test**

Run:

```bash
uv run python -m unittest tests.test_generation_contracts.GenerationContractTests.test_gitcode_pr_skill_uses_official_api_script -v
```

Expected: PASS.

- [ ] **Step 2: Run strict pyright on the new script**

Run:

```bash
bash scripts/run-skill-script-pyright.sh .codex/skills/managing-gitcode-prs/scripts/gitcode_pr_api.py
```

Expected: `0 errors`.

- [ ] **Step 3: Re-run the skill validator**

Run:

```bash
uv run python /Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py /Users/cdj/Projects/helix/.codex/skills/managing-gitcode-prs
```

Expected: `Skill is valid!`
