# Pi Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pi` as a supported agent backend without changing the existing CLI contract beyond allowing `--agent pi`.

**Architecture:** Follow the existing backend adapter pattern: keep CLI parsing and prompt construction backend-neutral, add a dedicated `PiRunner` for Pi command construction, and stage repository skills into a Pi-specific workspace location that the runner passes back to Pi explicitly.

**Tech Stack:** Python 3.11, `argparse`, `unittest`, `uv`, existing Triton agent CLI modules

---

### Task 1: Extend the CLI backend selection

**Files:**
- Modify: `src/triton_agent/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing parser tests**

Add parser coverage showing `gen-test`, `gen-bench`, and `optimize` accept `--agent pi`.

- [ ] **Step 2: Run parser-focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: FAIL because `pi` is not yet an allowed backend.

- [ ] **Step 3: Implement the parser and runner factory changes**

Update the `--agent` choice list and `create_runner(...)` so `pi` is treated like the other agent-backed commands.

- [ ] **Step 4: Re-run parser-focused tests**

Run: `uv run python -m unittest tests.test_cli.CliParserTests -v`
Expected: PASS

### Task 2: Add Pi runner and Pi skill staging

**Files:**
- Create: `src/triton_agent/pi_runner.py`
- Modify: `src/triton_agent/skills.py`
- Test: `tests/test_pi_runner.py`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Write the failing backend tests**

Add tests covering:
- interactive Pi command construction
- non-interactive Pi command construction
- verbose command logging
- unified process-runner dispatch
- `.pi/skills` copy staging and cleanup

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `uv run python -m unittest tests.test_pi_runner tests.test_skills -v`
Expected: FAIL because Pi runner and Pi skill staging do not exist yet.

- [ ] **Step 3: Implement the minimal backend support**

Add `PiRunner` with the same resume semantics as the other backends, then extend `SkillLinkManager` with Pi-specific staging under `.pi/skills` and explicit symlink rejection.

- [ ] **Step 4: Re-run targeted tests**

Run: `uv run python -m unittest tests.test_pi_runner tests.test_skills -v`
Expected: PASS

### Task 3: Update docs and run verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/2026-04-03-pi-backend.md`

- [ ] **Step 1: Update user-facing docs**

Document `pi` as a supported backend and describe the Pi-specific workspace skill staging behavior.

- [ ] **Step 2: Run full verification**

Run:
- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

Expected: all PASS
