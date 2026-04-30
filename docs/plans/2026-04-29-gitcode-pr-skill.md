# GitCode PR Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local Codex skill that helps create, list, and view GitCode pull requests through the official API.

**Architecture:** Keep all behavior in a new `.codex/skills/managing-gitcode-prs/` skill folder. Put the durable workflow in `SKILL.md`, keep UI metadata in `agents/openai.yaml`, add a deterministic repo-local API script under `scripts/`, and move endpoint detail into a small reference file so the trigger body stays concise.

**Tech Stack:** Markdown, YAML, Python validation scripts

---

### Task 1: Record the stable contract

**Files:**
- Create: `docs/specs/2026-04-29-gitcode-pr-skill-design.md`

- [ ] **Step 1: Write the design document**

Add a short design that fixes the scope to PR create/list/view tasks, requires `GC_TOKEN`, and keeps the skill repo-local under `.codex/skills/`.

- [ ] **Step 2: Review the design for scope drift**

Confirm the document does not expand into a general GitCode integration and does not require changes under `src/`.

### Task 2: Scaffold the skill directory

**Files:**
- Create: `.codex/skills/managing-gitcode-prs/SKILL.md`
- Create: `.codex/skills/managing-gitcode-prs/agents/openai.yaml`
- Create: `.codex/skills/managing-gitcode-prs/scripts/gitcode_pr_api.py`
- Create: `.codex/skills/managing-gitcode-prs/references/pr-command-reference.md`

- [ ] **Step 1: Initialize the skill skeleton**

Run:

```bash
python3 /Users/cdj/.codex/skills/.system/skill-creator/scripts/init_skill.py managing-gitcode-prs --path /Users/cdj/Projects/triton-agent/.codex/skills --resources references --interface display_name="GitCode PRs" --interface short_description="Create, list, and inspect GitCode PRs" --interface default_prompt="Use $managing-gitcode-prs to create or inspect a GitCode pull request for this repository."
```

Expected: the new skill directory is created with `SKILL.md`, `agents/openai.yaml`, and `references/`.

- [ ] **Step 2: Remove template-only guidance by replacing it with final content**

Replace the generated placeholder text so the skill folder only contains instructions relevant to GitCode PR work.

### Task 3: Author the skill content

**Files:**
- Modify: `.codex/skills/managing-gitcode-prs/SKILL.md`
- Modify: `.codex/skills/managing-gitcode-prs/agents/openai.yaml`
- Modify: `.codex/skills/managing-gitcode-prs/references/pr-command-reference.md`

- [ ] **Step 1: Write `SKILL.md`**

Document:
- the `GC_TOKEN` precheck
- when to use create, list, and view operations
- how to prefer current branch auto-detection but fall back to `--head`
- when to use `--fill`, `--draft`, `--json`, `--comments`, and `--time-format`
- how to report failures clearly when repo, branch detection, or HTTP access is unavailable

- [ ] **Step 2: Keep UI metadata aligned**

Ensure `agents/openai.yaml` has a concise display name, a 25-64 character short description, and a default prompt that explicitly mentions `$managing-gitcode-prs`.

- [ ] **Step 3: Add the command reference**

Capture the supported API-script command shapes and endpoint notes in `references/pr-command-reference.md` so the main skill file can stay compact.

### Task 4: Validate and inspect the result

**Files:**
- Modify: `.codex/skills/managing-gitcode-prs/` as needed

- [ ] **Step 1: Run the skill validator**

Run:

```bash
python3 /Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py /Users/cdj/Projects/triton-agent/.codex/skills/managing-gitcode-prs
```

Expected: `Skill is valid!`

- [ ] **Step 2: Review the generated diff**

Run:

```bash
git diff -- docs/specs/2026-04-29-gitcode-pr-skill-design.md docs/plans/2026-04-29-gitcode-pr-skill.md .codex/skills/managing-gitcode-prs
```

Expected: only the new design doc, plan, and PR-focused skill files are present.
