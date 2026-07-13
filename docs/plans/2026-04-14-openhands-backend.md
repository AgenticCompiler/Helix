# OpenHands Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-phase in-process `openhands` backend that supports existing non-interactive agent flows with workspace skill staging and friendly configuration errors.

**Architecture:** Keep the CLI surface change minimal by adding `openhands` to the existing backend selection flow. Implement an `OpenHandsRunner` adapter that maps the current `AgentRequest` and `AgentResult` contract onto OpenHands SDK objects, while extending `SkillLinkManager` with `.openhands/skills` staging and explicitly rejecting `--interact`. Keep workspace guidance behavior aligned with the other backends by staging skills only and by not auto-injecting the repository's own `AGENTS.md` into target workspaces.

**Tech Stack:** Python, `argparse`, `unittest`, OpenHands SDK, Markdown docs

---

### Task 1: Lock CLI And Factory Behavior In Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_backends_factory.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_backends_factory.py`

- [ ] **Step 1: Write the failing tests**

Add parser coverage for `--agent openhands` on agent-backed commands and factory coverage for `create_runner("openhands")`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_backends_factory.BackendFactoryTests -v`
Expected: FAIL because `openhands` is not accepted and the factory does not know the new backend.

- [ ] **Step 3: Write minimal implementation**

Update `src/helix/cli.py` and `src/helix/backends/factory.py` so `openhands` is accepted and maps to the new backend class.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_backends_factory.BackendFactoryTests -v`
Expected: PASS

### Task 2: Lock OpenHands Skill Staging In Tests

**Files:**
- Modify: `tests/test_skills.py`
- Modify: `src/helix/skills.py`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Write the failing tests**

Add coverage for `.openhands/skills` copy staging, symlink rejection, and cleanup semantics.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests -v`
Expected: FAIL because OpenHands staging does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Extend `SkillLinkManager` and `prepare_skills()` with OpenHands-specific staging behavior that mirrors the current backend rules.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests -v`
Expected: PASS

### Task 3: Build The OpenHands Runner With TDD

**Files:**
- Create: `src/helix/backends/openhands.py`
- Create: `tests/test_openhands_runner.py`
- Modify: `src/helix/commands/generation.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing tests**

Add runner coverage for:
- missing environment configuration
- unsupported `--interact`
- successful non-interactive execution via mocked SDK objects
- `resume()` preserving the shared continuation-prompt behavior

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest tests.test_openhands_runner -v`
Expected: FAIL because the OpenHands backend module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement `OpenHandsRunner`, add friendly OpenHands-specific setup errors, reject `--interact`, and wire the backend into generation and optimize command error handling. Add OpenHands runtime dependencies in `pyproject.toml`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_openhands_runner -v`
Expected: PASS

### Task 4: Run Full Verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused regression tests**

Run: `uv run python -m unittest tests.test_cli tests.test_backends_factory tests.test_skills tests.test_openhands_runner -v`
Expected: PASS

- [ ] **Step 2: Run lint**

Run: `uv run --group dev ruff check`
Expected: PASS

- [ ] **Step 3: Run static typing**

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
