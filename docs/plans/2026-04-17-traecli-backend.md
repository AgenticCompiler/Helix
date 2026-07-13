# TraeCLI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-phase `traecli` backend with interactive and non-interactive support, while refactoring backend skill staging to remove duplicated per-backend directory-copy logic.

**Architecture:** Keep the CLI surface minimal by adding `traecli` to the existing backend selection flow and implementing a dedicated `TraeCLIRunner` that reuses the shared `AgentRunner` execution and resume path. Refactor `SkillLinkManager` so backend-specific skill roots come from one mapping table, preserving the current copy-only, symlink-safe, cleanup-safe semantics while adding `.traecli/skills` as a new target.

**Tech Stack:** Python, argparse, pathlib, unittest, ruff, pyright, uv, Markdown docs

---

### Task 1: Lock CLI And Factory Wiring In Tests

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_backends_factory.py`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/backends/factory.py`
- Modify: `src/helix/backends/__init__.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_backends_factory.py`

- [ ] **Step 1: Write the failing parser and factory tests**

Add coverage that `--agent traecli` is accepted on all agent-backed commands and that `create_runner("traecli")` returns `TraeCLIRunner`.

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_backends_factory.BackendFactoryTests -v`
Expected: FAIL because `traecli` is not yet in `_AGENT_CHOICES` and the backend factory does not know the new runner.

- [ ] **Step 3: Write minimal implementation**

Update:
- `src/helix/cli.py` to include `traecli` in `_AGENT_CHOICES`
- `src/helix/backends/factory.py` to return `TraeCLIRunner`
- `src/helix/backends/__init__.py` to export `TraeCLIRunner`

- [ ] **Step 4: Re-run the focused tests**

Run: `uv run python -m unittest tests.test_cli.CliParserTests tests.test_backends_factory.BackendFactoryTests -v`
Expected: PASS

### Task 2: Refactor Skill Staging Around A Backend Directory Map

**Files:**
- Modify: `tests/test_skills.py`
- Modify: `src/helix/skills.py`
- Test: `tests/test_skills.py`

- [ ] **Step 1: Write the failing staging tests for the refactor target**

Reshape `tests/test_skills.py` so it asserts behavior through a backend-to-directory mapping instead of one near-duplicate test block per backend, and add `.traecli/skills` coverage.

Minimum coverage to keep:
- full-copy staging when the backend root is missing
- selective skill staging when only specific skill names are requested
- symlink rejection for the backend root or staged skill path
- cleanup removes only paths created by the current run
- existing user-owned directories are preserved

- [ ] **Step 2: Run the skill tests to verify they fail for `traecli` and the new abstraction**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests -v`
Expected: FAIL because there is no `traecli` backend mapping and the refactored expectations are not implemented yet.

- [ ] **Step 3: Write minimal implementation**

Refactor `src/helix/skills.py` to:
- introduce one backend-to-target mapping that includes:
  - `codex -> .codex/skills`
  - `opencode -> .opencode/skills`
  - `pi -> .pi/skills`
  - `claude -> .claude/skills`
  - `openhands -> .openhands/skills`
  - `traecli -> .traecli/skills`
- replace `prepare_codex_skills()`, `prepare_opencode_skills()`, `prepare_pi_skills()`, `prepare_claude_skills()`, and `prepare_openhands_skills()` with a shared helper that preserves the current copy and cleanup semantics
- keep `prepare_skills()` as the public dispatcher used by orchestration code

- [ ] **Step 4: Re-run the skill tests**

Run: `uv run python -m unittest tests.test_skills.SkillLinkManagerTests -v`
Expected: PASS

### Task 3: Build The TraeCLI Runner With TDD

**Files:**
- Create: `src/helix/backends/traecli.py`
- Create: `tests/test_traecli_runner.py`
- Modify: `src/helix/backends/factory.py`
- Modify: `src/helix/backends/__init__.py`
- Test: `tests/test_traecli_runner.py`

- [ ] **Step 1: Write the failing runner tests**

Add coverage for:
- interactive command construction: `traecli <prompt>`
- non-interactive command construction: `traecli --print --yolo <prompt>`
- `optimize --no-agent-session` being ignored
- shared process-runner dispatch through `AgentRunner.run()`
- verbose launch logging
- shared `resume()` preserving the continuation-prompt behavior already tested for other subprocess backends

- [ ] **Step 2: Run the TraeCLI runner tests to verify they fail**

Run: `uv run python -m unittest tests.test_traecli_runner -v`
Expected: FAIL because the backend module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `src/helix/backends/traecli.py` with a `TraeCLIRunner` that:
- subclasses `AgentRunner`
- uses `traecli` as the executable name by default
- returns `[self.executable, request.prompt]` for interactive mode
- returns `[self.executable, "--print", "--yolo", request.prompt]` for non-interactive mode
- does not add any synthetic no-session flag

- [ ] **Step 4: Re-run the TraeCLI runner tests**

Run: `uv run python -m unittest tests.test_traecli_runner -v`
Expected: PASS

### Task 4: Update User-Facing Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-04-17-traecli-backend-design.md`
- Modify: `docs/plans/2026-04-17-traecli-backend.md`

- [ ] **Step 1: Update backend lists in the README**

Add `traecli` anywhere the README currently enumerates `codex|opencode|pi|claude` or `codex|opencode|pi|claude|openhands` for agent-backed commands.

- [ ] **Step 2: Verify the docs stay aligned with implementation**

Double-check that the README and the spec both match the implemented behavior:
- interactive mode supported
- non-interactive mode uses `--print --yolo`
- skills stage to `.traecli/skills`
- `--no-agent-session` is ignored for TraeCLI in the first phase

### Task 5: Run Focused Regression And Full Verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused regression tests**

Run: `uv run python -m unittest tests.test_cli tests.test_backends_factory tests.test_skills tests.test_traecli_runner -v`
Expected: PASS

- [ ] **Step 2: Run lint**

Run: `uv run --group dev ruff check`
Expected: PASS

- [ ] **Step 3: Run static typing**

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 4: Run the full test suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 5: Fix only regressions caused by this change and re-run affected checks**

If any verification step fails, keep the scope limited to:
- `traecli` backend wiring
- skill staging refactor regressions
- README or test updates needed to match the implemented behavior
