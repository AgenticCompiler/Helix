# Create-PR Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local `create-pr` skill that requires a fresh topic branch, repository-wide validation, strict checking of all `skills/*/scripts/*.py` files, and then hands off final PR creation to the existing `managing-gitcode-prs` skill.

**Architecture:** Keep the new workflow entirely inside `.codex/skills/create-pr/` so the project CLI stays unchanged. Put the durable workflow in `SKILL.md`, keep UI metadata in `agents/openai.yaml`, and keep the skill self-contained so it only includes the repo-specific checks and handoff rules.

**Tech Stack:** Markdown, YAML, repo-local skill scaffolding scripts

---

### Task 1: Scaffold the repo-local skill

**Files:**
- Create: `.codex/skills/create-pr/SKILL.md`
- Create: `.codex/skills/create-pr/agents/openai.yaml`

- [ ] **Step 1: Initialize the skill skeleton**

Run:

```bash
python3 /Users/cdj/.codex/skills/.system/skill-creator/scripts/init_skill.py create-pr --path /Users/cdj/Projects/helix/.codex/skills --resources references --interface display_name="Create PR" --interface short_description="Run repo checks and open a GitCode PR" --interface default_prompt="Use $create-pr to validate this repository, push a fresh branch, and open a pull request."
```

Expected: `.codex/skills/create-pr/` exists with `SKILL.md`, `agents/openai.yaml`, and `references/`.

- [ ] **Step 2: Verify the scaffolded files exist**

Run:

```bash
find .codex/skills/create-pr -maxdepth 2 -type f | sort
```

Expected: the output lists `SKILL.md`, `agents/openai.yaml`, and at least one file under `references/`.

### Task 2: Replace the template with the final workflow

**Files:**
- Modify: `.codex/skills/create-pr/SKILL.md`

- [ ] **Step 1: Write the main skill contract**

Replace the template body in `.codex/skills/create-pr/SKILL.md` so it:

- triggers for creating a new PR in this repository
- requires a newly created topic branch but no forced branch prefix
- requires these commands before PR creation:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

- requires strict checking of every skill-side script with:

```bash
find skills -path '*/scripts/*.py' | sort | while IFS= read -r file; do
  bash scripts/run-skill-script-pyright.sh "$file"
done
```

- requires the new branch to be pushed before PR creation
- explicitly delegates the final PR creation step to `$managing-gitcode-prs`
- avoids generic git command walkthroughs and repeated PR API examples

### Task 3: Align the repo-local UI metadata

**Files:**
- Modify: `.codex/skills/create-pr/agents/openai.yaml`

- [ ] **Step 1: Confirm the generated UI metadata matches the final skill**

Ensure `agents/openai.yaml` keeps:

```yaml
interface:
  display_name: "Create PR"
  short_description: "Run repo checks and open a GitCode PR"
  default_prompt: "Use $create-pr to validate this repository, push a fresh branch, and open a pull request."
```

Expected: the metadata names the repo-local skill clearly and the default prompt mentions `$create-pr`.

### Task 4: Validate and inspect the finished skill

**Files:**
- Modify: `.codex/skills/create-pr/` as needed

- [ ] **Step 1: Run the skill validator**

Run:

```bash
python3 /Users/cdj/.codex/skills/.system/skill-creator/scripts/quick_validate.py /Users/cdj/Projects/helix/.codex/skills/create-pr
```

Expected: `Skill is valid!`

- [ ] **Step 2: Review the final diff**

Run:

```bash
git diff -- docs/specs/2026-06-03-create-pr-skill-design.md docs/plans/2026-06-03-create-pr-skill.md .codex/skills/create-pr
```

Expected: the diff only contains the new spec, plan, and repo-local `create-pr` skill files, with the workflow constraints matching the approved design.
