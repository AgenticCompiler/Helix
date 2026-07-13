# Skill Staging Rule Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize all subcommand skill-staging rules in one table and resolve them through a shared helper.

**Architecture:** Add a small shared staging module that owns the rule table and the `+`/`-`/`*` resolution logic. Generation, convert, and optimize orchestration should ask that helper for `staged_skill_names` and `staged_skill_sources` instead of hard-coding local lists. Keep `SkillLinkManager` focused on copying and cleanup only.

**Tech Stack:** Python 3.12, `unittest`, existing orchestration modules under `src/helix/`.

---

### Task 1: Add the shared staging resolver

**Files:**
- Create: `src/helix/skill_staging.py`
- Create: `tests/test_skill_staging.py`

- [ ] **Step 1: Write the failing test**

```python
from helix.models import CommandKind
from helix.skill_staging import resolve_staged_skills

def test_gen_eval_rule_expands_to_the_expected_set():
    names, sources = resolve_staged_skills(CommandKind.GEN_EVAL)
    assert names == (
        "triton-npu-gen-eval-suite",
        "triton-npu-gen-test",
        "triton-npu-gen-bench",
        "triton-npu-run-eval",
    )
    assert sources is None

def test_plus_minus_directives_are_applied_in_order():
    assert resolve_stage_directives(("*", "-b", "+c"), ("a", "b")) == ("a", "c")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_skill_staging -v`
Expected: FAIL because the shared resolver does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass
from helix.models import CommandKind

@dataclass(frozen=True)
class StageRule:
    directives: tuple[str, ...]
    skill_sources: dict[str, str] | None = None

STAGE_RULES = {
    CommandKind.GEN_EVAL: StageRule(
        directives=(
            "+triton-npu-gen-eval-suite",
            "+triton-npu-gen-test",
            "+triton-npu-gen-bench",
            "+triton-npu-run-eval",
        ),
    ),
}

def resolve_staged_skills(command_kind: CommandKind) -> tuple[tuple[str, ...] | None, dict[str, str] | None]:
    rule = STAGE_RULES[command_kind]
    return tuple(name[1:] for name in rule.directives if name.startswith("+")), rule.skill_sources
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python -m unittest tests.test_skill_staging -v`
Expected: PASS.

### Task 2: Wire orchestration through the shared table

**Files:**
- Modify: `src/helix/generation/orchestration.py`
- Modify: `src/helix/convert/orchestration.py`
- Modify: `src/helix/optimize/orchestration.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_generation_request_reads_stage_rules():
    request = build_generation_request(...)
    assert request.staged_skill_names == (...)

def test_build_optimize_request_preserves_skill_source_alias():
    request = build_optimize_request(...)
    assert request.staged_skill_sources == {"triton-npu-optimize-knowledge": "triton-npu-optimize-knowledge-v2"}
```

- [ ] **Step 2: Run the focused tests**

Run: `uv run python -m unittest tests.test_generation_commands tests.test_convert_commands tests.test_optimize_runtime -v`
Expected: FAIL because the orchestration modules still hard-code their local skill lists.

- [ ] **Step 3: Replace local staging constants with resolver calls**

```python
from helix.skill_staging import resolve_staged_skills

staged_skill_names, staged_skill_sources = resolve_staged_skills(command_kind, options=options)
```

- [ ] **Step 4: Run the focused tests again**

Run: `uv run python -m unittest tests.test_generation_commands tests.test_convert_commands tests.test_optimize_runtime -v`
Expected: PASS.

### Task 3: Update staging tests and docs

**Files:**
- Modify: `tests/test_skills.py`
- Modify: `docs/specs/2026-05-11-skill-staging-rule-table-design.md`
- Modify: `docs/plans/2026-05-11-skill-staging-rule-table-plan.md`

- [ ] **Step 1: Add coverage for the directive syntax**

```python
def test_stage_rule_language_supports_full_copy_plus_exclusions():
    ...
```

- [ ] **Step 2: Add coverage that current backend copy behavior is unchanged**

```python
def test_backend_copy_semantics_remain_conservative():
    ...
```

- [ ] **Step 3: Run the repository tests that cover the touched paths**

Run: `uv run python -m unittest tests.test_skill_staging tests.test_generation_commands tests.test_convert_commands tests.test_optimize_runtime tests.test_skills -v`
Expected: PASS.

