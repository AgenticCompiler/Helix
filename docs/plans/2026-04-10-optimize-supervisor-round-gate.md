# Optimize Supervisor Round Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `optimize` into an explicit worker-round plus supervisor-gate loop that enforces the optimize workflow, blocks missing evidence, and produces deterministic next-round handoff artifacts.

**Architecture:** Keep orchestration in the CLI/runtime layer and keep optimization behavior in skills. Add a small structured round contract and gate result model under `src/triton_agent/optimize/`, render shared guidance plus role-specific briefs in the workspace, and upgrade the optimize supervisor from stall recovery into a round-aware controller that launches fresh worker and supervisor invocations.

**Tech Stack:** Python `dataclasses`, `pathlib`, JSON, existing optimize runtime/backends, Python `unittest`

---

## File Structure

**New files**

- `src/triton_agent/optimize/round_contract.py`
  Round artifact discovery, `round-state.json` parsing, and comparable validation helpers.
- `src/triton_agent/optimize/gate.py`
  Gate decision enums/models plus supervisor-facing round inspection results.
- `tests/test_optimize_round_contract.py`
  Focused unit tests for round artifact and `round-state.json` handling.
- `tests/test_optimize_gate.py`
  Decision-path tests for `pass-continue`, `pass-stop`, `revise-metadata`, `revise-required`, and `hard-fail`.
- `skills/optimize-supervisor/SKILL.md`
  Audit-oriented optimize supervisor skill.

**Existing files to modify**

- `src/triton_agent/models.py`
  Extend `AgentRequest` with explicit optimize role/session-loop fields.
- `src/triton_agent/optimize/models.py`
  Add round/gate/guidance-related dataclasses.
- `src/triton_agent/prompts.py`
  Split optimize worker and supervisor prompt builders.
- `src/triton_agent/optimize_guidance.py`
  Render role-neutral shared guidance plus role briefs instead of one worker-only guidance file.
- `src/triton_agent/optimize/runtime.py`
  Prepare shared skills/guidance and run the worker-supervisor loop.
- `src/triton_agent/supervisor.py`
  Replace the current stall/min-round-only logic with round gate orchestration.
- `src/triton_agent/backends/codex.py`
  Preserve fresh invocation behavior and explicit prompt routing for worker versus supervisor runs.
- `src/triton_agent/backends/opencode.py`
- `src/triton_agent/backends/claude.py`
- `src/triton_agent/backends/pi.py`
  Mirror any `AgentRequest` field additions that affect optimize launch behavior.
- `tests/test_supervisor.py`
  Update supervisor tests to the new loop behavior.
- `tests/test_optimize_guidance.py`
  Cover shared guidance and role-brief rendering.
- `tests/test_skills.py`
  Cover staging of the new supervisor skill.
- `README.md`
  Document the round gate behavior at a workflow level.
- `AGENTS.md`
  Update stable project rules only if the new supervisor role changes durable workflow expectations.

### Task 1: Add The Round Contract And Gate Models

**Files:**
- Create: `src/triton_agent/optimize/round_contract.py`
- Create: `src/triton_agent/optimize/gate.py`
- Modify: `src/triton_agent/optimize/models.py`
- Create: `tests/test_optimize_round_contract.py`
- Create: `tests/test_optimize_gate.py`

- [ ] **Step 1: Write the failing round-contract tests**

Add tests that lock the required artifact behavior before implementation:

```python
def test_load_round_state_requires_core_fields(self) -> None:
    round_dir = workspace / "opt-round-1"
    round_dir.mkdir()
    (round_dir / "round-state.json").write_text('{"round": "opt-round-1"}', encoding="utf-8")
    with self.assertRaises(ValueError):
        load_round_state(round_dir)

def test_inspect_round_artifacts_flags_missing_summary(self) -> None:
    result = inspect_round_artifacts(workspace / "opt-round-1")
    self.assertIn("missing summary.md", result.issues)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_round_contract tests.test_optimize_gate -v`
Expected: FAIL because the round contract helpers and gate models do not exist yet

- [ ] **Step 3: Implement the minimal round contract and gate types**

Define focused dataclasses and helpers instead of putting parsing logic into `supervisor.py`:

```python
@dataclass(frozen=True)
class RoundState:
    round_name: str
    parent_round: str
    hypothesis: str
    evidence_sources: tuple[str, ...]
    correctness_status: str
    benchmark_status: str
    perf_artifact: str
    summary_path: str
    opt_note_updated: bool
    next_recommendation: str
```

```python
class GateDecision(str, Enum):
    PASS_CONTINUE = "pass-continue"
    PASS_STOP = "pass-stop"
    REVISE_METADATA = "revise-metadata"
    REVISE_REQUIRED = "revise-required"
    HARD_FAIL = "hard-fail"
```

Keep the parser strict and return short actionable issue messages instead of traceback-shaped errors.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_round_contract tests.test_optimize_gate -v`
Expected: PASS

- [ ] **Step 5: Commit the contract layer**

```bash
git add src/triton_agent/optimize/round_contract.py src/triton_agent/optimize/gate.py src/triton_agent/optimize/models.py tests/test_optimize_round_contract.py tests/test_optimize_gate.py
git commit -m "feat: add optimize round gate contract models"
```

### Task 2: Split Optimize Prompts And Request Metadata By Role

**Files:**
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/backends/codex.py`
- Modify: `src/triton_agent/backends/opencode.py`
- Modify: `src/triton_agent/backends/claude.py`
- Modify: `src/triton_agent/backends/pi.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Write the failing prompt and request-shape tests**

Add tests proving worker and supervisor invocations are distinct:

```python
def test_build_optimize_worker_prompt_mentions_single_round_boundary(self) -> None:
    prompt = build_optimize_worker_prompt(...)
    self.assertIn("exactly one round", prompt)

def test_build_optimize_supervisor_prompt_mentions_audit_role(self) -> None:
    prompt = build_optimize_supervisor_prompt(...)
    self.assertIn("audit and handoff pass", prompt)
```

Also add a supervisor test that asserts a resumed worker run does not silently reuse a supervisor role.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_supervisor -v`
Expected: FAIL because optimize prompts and requests still model a single undifferentiated optimize role

- [ ] **Step 3: Implement the minimal request and prompt split**

Extend `AgentRequest` with explicit optimize-role metadata such as:

```python
@dataclass
class AgentRequest:
    ...
    optimize_role: str | None = None
    round_brief_path: Path | None = None
    supervisor_report_path: Path | None = None
```

Add dedicated builders instead of overloading one giant optimize prompt branch:

```python
def build_optimize_worker_prompt(...) -> str: ...
def build_optimize_supervisor_prompt(...) -> str: ...
```

Keep backend launch behavior simple: prompts drive the role, and optimize invocations stay ephemeral whenever session reuse could blur worker/supervisor boundaries.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_supervisor -v`
Expected: PASS

- [ ] **Step 5: Commit the prompt split**

```bash
git add src/triton_agent/models.py src/triton_agent/prompts.py src/triton_agent/backends/codex.py src/triton_agent/backends/opencode.py src/triton_agent/backends/claude.py src/triton_agent/backends/pi.py tests/test_supervisor.py
git commit -m "feat: add optimize worker and supervisor prompts"
```

### Task 3: Render Shared Guidance And Role Briefs

**Files:**
- Modify: `src/triton_agent/optimize_guidance.py`
- Modify: `src/triton_agent/optimize/runtime.py`
- Modify: `tests/test_optimize_guidance.py`

- [ ] **Step 1: Write the failing guidance tests**

Add tests that pin the new guidance layout:

```python
def test_prepare_writes_shared_guidance_and_role_briefs(self) -> None:
    state = manager.prepare(...)
    self.assertTrue((workdir / ".triton-agent/roles/optimize-worker.md").exists())
    self.assertTrue((workdir / ".triton-agent/roles/optimize-supervisor.md").exists())

def test_cleanup_removes_role_briefs_and_restores_original_guidance(self) -> None:
    warnings = manager.cleanup(state)
    self.assertFalse((workdir / ".triton-agent/roles/optimize-worker.md").exists())
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_optimize_guidance -v`
Expected: FAIL because the guidance manager still writes a single worker-oriented top-level file

- [ ] **Step 3: Implement shared guidance plus role-brief rendering**

Refactor `OptimizeGuidanceManager` so `prepare()` writes:

- one shared top-level `AGENTS.md` or `CLAUDE.md`
- `.triton-agent/roles/optimize-worker.md`
- `.triton-agent/roles/optimize-supervisor.md`
- a writable location for `.triton-agent/round-brief.md`
- a writable location for `.triton-agent/supervisor-report.md`

Keep the shared top-level guidance strictly role-neutral because some backends may treat workspace guidance as memory-file context with higher priority than ordinary file reads. Put role assignment in the launch prompt, not only in the role brief files.

Keep cleanup conservative: remove only files created by the current run and restore backed-up top-level guidance exactly as today.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_optimize_guidance -v`
Expected: PASS

- [ ] **Step 5: Commit the guidance layer**

```bash
git add src/triton_agent/optimize_guidance.py src/triton_agent/optimize/runtime.py tests/test_optimize_guidance.py
git commit -m "feat: add optimize role guidance briefs"
```

### Task 4: Upgrade The Supervisor Into A Round Gate Controller

**Files:**
- Modify: `src/triton_agent/supervisor.py`
- Modify: `src/triton_agent/optimize/runtime.py`
- Modify: `tests/test_supervisor.py`
- Modify: `tests/test_optimize_gate.py`

- [ ] **Step 1: Write the failing orchestration tests**

Add tests that lock the new decision flow:

```python
def test_supervisor_runs_worker_then_supervisor_then_stops_on_pass_stop(self) -> None:
    ...
    self.assertEqual(events, ["worker-run", "supervisor-run"])

def test_supervisor_relaunches_worker_with_repair_brief_on_revise_required(self) -> None:
    ...
    self.assertIn("required evidence", next_worker_request.prompt)
```

Also keep a focused regression test for existing stall recovery only if that behavior still belongs in the new controller. If not, delete the obsolete stall-path tests rather than forcing the new design to mimic the old one.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_supervisor tests.test_optimize_gate -v`
Expected: FAIL because `OptimizeSupervisor` still only handles stalled runs and `min_rounds`

- [ ] **Step 3: Implement the minimal worker-supervisor loop**

Refactor `OptimizeSupervisor` around explicit rounds:

```python
class OptimizeSupervisor:
    def run(self, runner: SupportsOptimizeLoop, request: AgentRequest) -> AgentResult:
        round_index = 1
        while True:
            worker_result = runner.run_worker(...)
            gate_result = runner.run_supervisor(...)
            decision = parse_gate_result(...)
            ...
```

Keep the first version intentionally narrow:

- supervisor may apply metadata-only repair
- `revise-required` relaunches a worker repair pass
- `hard-fail` stops immediately
- `pass-stop` returns success
- `pass-continue` writes the next-round brief and advances the loop

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_supervisor tests.test_optimize_gate -v`
Expected: PASS

- [ ] **Step 5: Commit the orchestration layer**

```bash
git add src/triton_agent/supervisor.py src/triton_agent/optimize/runtime.py tests/test_supervisor.py tests/test_optimize_gate.py
git commit -m "feat: add optimize supervisor round gate loop"
```

### Task 5: Add The Supervisor Skill And Wire It Into Staging

**Files:**
- Create: `skills/optimize-supervisor/SKILL.md`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/runtime.py`
- Modify: `tests/test_skills.py`

- [ ] **Step 1: Write the failing skill-staging tests**

Add a test that proves optimize orchestration stages the new supervisor skill alongside existing optimize-related skills:

```python
def test_optimize_runtime_stages_supervisor_skill(self) -> None:
    ...
    self.assertTrue((target_skills_dir / "optimize-supervisor").exists())
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run python -m unittest tests.test_skills -v`
Expected: FAIL because the new skill does not exist and optimize staging does not require it

- [ ] **Step 3: Implement the minimal skill wiring**

Write `skills/optimize-supervisor/SKILL.md` as an audit-first companion to `skills/optimize/SKILL.md`:

- read the optimize workflow and current round artifacts first
- decide against the five gate states
- repair only metadata derived from existing facts
- emit a structured gate result and next-round brief

Update optimize runtime staging so the workspace contains the supervisor skill whenever optimize orchestration runs.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `uv run python -m unittest tests.test_skills -v`
Expected: PASS

- [ ] **Step 5: Commit the skill layer**

```bash
git add skills/optimize-supervisor/SKILL.md src/triton_agent/models.py src/triton_agent/optimize/runtime.py tests/test_skills.py
git commit -m "feat: add optimize supervisor skill"
```

### Task 6: Update Docs And Run Full Verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md` only if durable workflow rules changed
- Modify: `docs/specs/2026-04-10-optimize-supervisor-round-gate-design.md` only if implementation discoveries require spec correction

- [ ] **Step 1: Update workflow-level docs**

Document only user-visible workflow semantics:

- optimize now advances in explicit worker rounds
- each round is audited before the next round may start
- supervisor may repair metadata but may not invent missing evidence
- optimize uses fresh agent invocations per role to avoid role leakage

- [ ] **Step 2: Run repository verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

- [ ] **Step 3: Commit the docs and verification pass**

```bash
git add README.md AGENTS.md docs/specs/2026-04-10-optimize-supervisor-round-gate-design.md
git commit -m "docs: describe optimize supervisor round gate workflow"
```

## Notes For Execution

- Prefer implementing Tasks 1 through 4 in order; later tasks depend on those interfaces staying stable.
- Keep the MVP narrow: metadata repair only, no supervisor-driven experiment reruns.
- If implementation reveals that existing stall recovery no longer belongs in `OptimizeSupervisor`, remove it cleanly instead of mixing two supervision models into one class.
- If `AGENTS.md` does not need a durable rule change, leave it untouched and keep the new role behavior in runtime-generated guidance plus skills.
