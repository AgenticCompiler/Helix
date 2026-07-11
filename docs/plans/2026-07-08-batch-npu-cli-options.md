# Batch NPU CLI Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--npu-devices` and `--workers-per-npu` as first-class batch-affinity CLI options, keep the legacy `HELIX_BATCH_*` variables as parser-level fallback, and pass normalized values explicitly into managed and standalone run-eval MCP startup.

**Architecture:** Normalize batch-affinity inputs once in the CLI layer with `option > env`, then thread those explicit values through command option models, batch-affinity helpers, and managed MCP startup. Keep standalone `run-eval-mcp-server` executable-style behavior by letting MCP startup helpers accept explicit values but fall back to environment variables when omitted.

**Tech Stack:** Python 3.12, argparse, dataclasses, pytest/unittest, uv-managed CLI runtime

---

### Task 1: Add parser coverage for new CLI options and compatibility fallback

**Files:**
- Modify: `src/helix/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing parser tests**

```python
def test_optimize_batch_accepts_batch_affinity_cli_options(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "optimize-batch",
            "-i",
            "kernels",
            "--npu-devices",
            "0,1",
            "--workers-per-npu",
            "2",
        ]
    )
    self.assertEqual(args.npu_devices, "0,1")
    self.assertEqual(args.workers_per_npu, "2")


def test_run_eval_mcp_server_accepts_batch_affinity_cli_options(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-eval-mcp-server",
            "--npu-devices",
            "0,1",
            "--workers-per-npu",
            "2",
        ]
    )
    self.assertEqual(args.npu_devices, "0,1")
    self.assertEqual(args.workers_per_npu, "2")


def test_main_prefers_explicit_batch_affinity_options_over_legacy_env(self) -> None:
    captured: dict[str, object] = {}

    def _fake_handle_optimize_batch(parser, args):
        del parser
        captured["npu_devices"] = args.npu_devices
        captured["workers_per_npu"] = args.workers_per_npu
        return 0

    with (
        patch.dict(
            os.environ,
            {
                "HELIX_BATCH_NPU_DEVICES": "4,5",
                "HELIX_BATCH_WORKERS_PER_NPU": "9",
            },
            clear=False,
        ),
        patch("helix.cli.handle_optimize_batch", side_effect=_fake_handle_optimize_batch),
    ):
        exit_code = main(
            [
                "optimize-batch",
                "-i",
                "kernels",
                "--npu-devices",
                "0,1",
                "--workers-per-npu",
                "2",
            ]
        )

    self.assertEqual(exit_code, 0)
    self.assertEqual(captured["npu_devices"], "0,1")
    self.assertEqual(captured["workers_per_npu"], "2")


def test_main_uses_legacy_env_when_batch_affinity_options_are_omitted(self) -> None:
    captured: dict[str, object] = {}

    def _fake_handle_optimize_batch(parser, args):
        del parser
        captured["npu_devices"] = args.npu_devices
        captured["workers_per_npu"] = args.workers_per_npu
        return 0

    with (
        patch.dict(
            os.environ,
            {
                "HELIX_BATCH_NPU_DEVICES": "0,1",
                "HELIX_BATCH_WORKERS_PER_NPU": "2",
            },
            clear=False,
        ),
        patch("helix.cli.handle_optimize_batch", side_effect=_fake_handle_optimize_batch),
    ):
        exit_code = main(["optimize-batch", "-i", "kernels"])

    self.assertEqual(exit_code, 0)
    self.assertEqual(captured["npu_devices"], "0,1")
    self.assertEqual(captured["workers_per_npu"], "2")
```

- [ ] **Step 2: Run the parser tests to verify they fail**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k "batch_affinity or run_eval_mcp_server_accepts_batch_affinity_cli_options"
```

Expected: FAIL because `--workers-per-npu` is not defined and the parsed args are not normalized from legacy env.

- [ ] **Step 3: Add the CLI arguments and one normalization helper**

```python
def _normalize_batch_affinity_args(args: argparse.Namespace) -> None:
    if not hasattr(args, "npu_devices"):
        return
    if getattr(args, "npu_devices", None) is None:
        args.npu_devices = os.environ.get("HELIX_BATCH_NPU_DEVICES")
    if hasattr(args, "workers_per_npu") and getattr(args, "workers_per_npu", None) is None:
        args.workers_per_npu = os.environ.get("HELIX_BATCH_WORKERS_PER_NPU")
```

```python
if spec.has_batch_affinity:
    subparser.add_argument("--npu-devices")
    subparser.add_argument("--workers-per-npu")
```

```python
args = parser.parse_args(normalized_argv)
_normalize_batch_affinity_args(args)
```

- [ ] **Step 4: Run the parser tests to verify they pass**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k "batch_affinity or run_eval_mcp_server_accepts_batch_affinity_cli_options"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/helix/cli.py tests/test_cli.py
git commit -m "feat: add batch affinity cli option parsing"
```

### Task 2: Refactor batch-affinity helpers to accept explicit values

**Files:**
- Modify: `src/helix/batch/affinity.py`
- Modify: `src/helix/commands/generation.py`
- Modify: `src/helix/commands/convert.py`
- Modify: `src/helix/commands/optimize.py`
- Modify: `src/helix/generation/models.py`
- Modify: `src/helix/convert/models.py`
- Modify: `src/helix/optimize/models.py`
- Test: `tests/test_npu_affinity.py`

- [ ] **Step 1: Write the failing helper tests for explicit raw inputs**

```python
def test_batch_slots_from_raw_values_returns_none_when_devices_unset(self) -> None:
    self.assertIsNone(configured_batch_npu_slots(None, None))


def test_batch_slots_from_raw_values_expands_devices_by_workers(self) -> None:
    self.assertEqual(
        configured_batch_npu_slots("0,1", "2"),
        ("0", "0", "1", "1"),
    )


def test_effective_capacity_uses_explicit_raw_inputs(self) -> None:
    self.assertEqual(effective_batch_affinity_capacity("0,1", "3"), 6)


def test_resolve_batch_concurrency_expands_max_from_explicit_inputs(self) -> None:
    self.assertEqual(resolve_batch_concurrency("max", "0,1", "2"), 4)


def test_validate_capacity_uses_explicit_workers_value(self) -> None:
    with self.assertRaisesRegex(ValueError, "HELIX_BATCH_WORKERS_PER_NPU"):
        validate_batch_affinity_capacity(("0", "1"), max_concurrency=5, workers_per_npu_raw="2")
```

- [ ] **Step 2: Run the affinity tests to verify they fail**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_npu_affinity.py
```

Expected: FAIL because the helpers currently only read process environment.

- [ ] **Step 3: Refactor the helpers and thread explicit option values into command options**

```python
def configured_batch_npu_slots(
    npu_devices_raw: str | None,
    workers_per_npu_raw: str | None,
) -> tuple[str, ...] | None:
    devices = parse_batch_npu_devices(npu_devices_raw)
    if devices is None:
        return None
    workers = parse_batch_workers_per_npu(workers_per_npu_raw)
    slots: list[str] = []
    for device in devices:
        slots.extend([device] * workers)
    return tuple(slots)
```

```python
@dataclass(frozen=True)
class ConvertOptions:
    ...
    npu_devices: str | None = None
    workers_per_npu: str | None = None
```

```python
return ConvertOptions(
    ...
    npu_devices=getattr(args, "npu_devices", None),
    workers_per_npu=getattr(args, "workers_per_npu", None),
)
```

```python
max_concurrency = resolve_batch_concurrency(
    args.concurrency,
    getattr(args, "npu_devices", None),
    getattr(args, "workers_per_npu", None),
)
```

- [ ] **Step 4: Run the focused affinity and batch tests to verify they pass**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings \
  tests/test_npu_affinity.py \
  tests/test_generation_batch.py \
  tests/test_convert_commands.py \
  tests/test_optimize_runtime.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  src/helix/batch/affinity.py \
  src/helix/commands/generation.py \
  src/helix/commands/convert.py \
  src/helix/commands/optimize.py \
  src/helix/generation/models.py \
  src/helix/convert/models.py \
  src/helix/optimize/models.py \
  tests/test_npu_affinity.py
git commit -m "feat: thread explicit batch affinity options"
```

### Task 3: Pass normalized batch-affinity values into managed MCP startup

**Files:**
- Modify: `src/helix/models.py`
- Modify: `src/helix/generation/orchestration.py`
- Modify: `src/helix/convert/orchestration.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/eval/mcp.py`
- Modify: `src/helix/backends/codex.py`
- Modify: `src/helix/backends/claude.py`
- Modify: `src/helix/backends/opencode.py`
- Test: `tests/test_cli.py`
- Test: backend-managed-MCP tests if present

- [ ] **Step 1: Write the failing managed MCP propagation tests**

```python
def test_handle_run_eval_mcp_server_passes_explicit_batch_affinity_values(self) -> None:
    args = SimpleNamespace(port=1234, npu_devices="0,1", workers_per_npu="2")
    with patch("helix.commands.mcp_server.serve_http_server_forever", return_value=0) as mocked:
        exit_code = handle_run_eval_mcp_server(argparse.ArgumentParser(), args)
    self.assertEqual(exit_code, 0)
    mocked.assert_called_once_with(port=1234, npu_devices="0,1", workers_per_npu="2")
```

```python
def test_build_generation_request_carries_batch_affinity_values(self) -> None:
    options = GenerationOptions(
        interact=False,
        verbose=False,
        stream_output=False,
        force_overwrite=False,
        agent_name="codex",
        remote=None,
        remote_workdir=None,
        min_rounds=None,
        continue_optimize=False,
        output=None,
        test_mode=None,
        bench_mode=None,
        npu_devices="0,1",
        workers_per_npu="2",
        enable_mcp=True,
    )
    request = build_generation_request(
        CommandKind.GEN_EVAL,
        Path("kernel.py"),
        Path("kernel.py"),
        Path.cwd(),
        options,
    )
    self.assertEqual(request.npu_devices, "0,1")
    self.assertEqual(request.workers_per_npu, "2")
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k "run_eval_mcp_server_passes_explicit_batch_affinity_values or carries_batch_affinity_values"
```

Expected: FAIL because the request model and MCP startup path do not yet accept explicit batch-affinity values.

- [ ] **Step 3: Extend request/context models and managed MCP startup signatures**

```python
@dataclass
class AgentRequest:
    ...
    npu_devices: str | None = None
    workers_per_npu: str | None = None
```

```python
return AgentRequest(
    ...
    npu_devices=options.npu_devices,
    workers_per_npu=options.workers_per_npu,
)
```

```python
@dataclass
class _ManagedMcpScopeState:
    server: RunningHttpMCPServer | None = None
    ref_count: int = 0
    npu_devices: str | None = None
    workers_per_npu: str | None = None
```

```python
def managed_mcp_scope(
    *,
    npu_devices: str | None = None,
    workers_per_npu: str | None = None,
) -> Iterator[None]:
    ...
```

```python
state.server = start_http_server(
    npu_devices=state.npu_devices,
    workers_per_npu=state.workers_per_npu,
)
```

- [ ] **Step 4: Run the focused managed MCP tests to verify they pass**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k "run_eval_mcp_server_passes_explicit_batch_affinity_values or carries_batch_affinity_values"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  src/helix/models.py \
  src/helix/generation/orchestration.py \
  src/helix/convert/orchestration.py \
  src/helix/optimize/orchestration.py \
  src/helix/eval/mcp.py \
  src/helix/backends/codex.py \
  src/helix/backends/claude.py \
  src/helix/backends/opencode.py \
  src/helix/commands/mcp_server.py \
  tests/test_cli.py
git commit -m "feat: pass batch affinity config into managed mcp"
```

### Task 4: Update run-eval MCP server startup to prefer explicit args and fall back to env

**Files:**
- Modify: `src/helix/eval/mcp_server.py`
- Modify: `src/helix/commands/mcp_server.py`
- Test: `tests/test_run_eval_mcp_server.py`

- [ ] **Step 1: Write the failing startup tests**

```python
def test_configured_slot_pool_uses_explicit_devices_over_env(self) -> None:
    with patch.dict(
        os.environ,
        {
            "HELIX_BATCH_NPU_DEVICES": "4,5",
            "HELIX_BATCH_WORKERS_PER_NPU": "9",
        },
        clear=False,
    ):
        pool = module.configured_slot_pool(npu_devices="0,1", workers_per_npu="2")

    seen_devices: list[str] = []
    with pool.acquire() as first:
        seen_devices.append(first)
        with pool.acquire() as second:
            seen_devices.append(second)

    self.assertEqual(seen_devices, ["0", "1"])


def test_configured_slot_pool_falls_back_to_env_when_args_omitted(self) -> None:
    with patch.dict(
        os.environ,
        {
            "HELIX_BATCH_NPU_DEVICES": "0,1",
            "HELIX_BATCH_WORKERS_PER_NPU": "3",
        },
        clear=False,
    ):
        pool = module.configured_slot_pool()

    seen_devices: list[str] = []
    with pool.acquire() as first:
        seen_devices.append(first)
        with pool.acquire() as second:
            seen_devices.append(second)

    self.assertEqual(seen_devices, ["0", "1"])
```

- [ ] **Step 2: Run the MCP server tests to verify they fail**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_run_eval_mcp_server.py
```

Expected: FAIL because `configured_slot_pool()` and startup helpers do not yet accept explicit batch-affinity arguments.

- [ ] **Step 3: Add optional explicit parameters with env fallback**

```python
def configured_slot_pool(
    *,
    npu_devices: str | None = None,
    workers_per_npu: str | None = None,
) -> NpuDevicePool:
    raw_devices = npu_devices
    if raw_devices is None:
        raw_devices = os.environ.get("HELIX_BATCH_NPU_DEVICES")
    normalized_devices = raw_devices.strip() if raw_devices is not None else ""
    devices = parse_batch_npu_devices(normalized_devices or "0")
    if devices is None:
        raise ValueError("Managed run-eval MCP server resolved no NPU devices.")
    raw_workers = workers_per_npu
    if raw_workers is None:
        raw_workers = os.environ.get("HELIX_BATCH_WORKERS_PER_NPU")
    parsed_workers = parse_batch_workers_per_npu(
        raw_workers if raw_workers and raw_workers.strip() else None
    )
    return NpuDevicePool(build_slot_pool(",".join(devices), parsed_workers))
```

```python
def start_http_server(
    *,
    port: int = 0,
    npu_devices: str | None = None,
    workers_per_npu: str | None = None,
) -> RunningHttpMCPServer:
    server = create_server(
        slot_pool=configured_slot_pool(
            npu_devices=npu_devices,
            workers_per_npu=workers_per_npu,
        )
    )
```

- [ ] **Step 4: Run the MCP server tests to verify they pass**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_run_eval_mcp_server.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/helix/eval/mcp_server.py src/helix/commands/mcp_server.py tests/test_run_eval_mcp_server.py
git commit -m "feat: support explicit batch affinity for run-eval mcp startup"
```

### Task 5: Refresh docs and run repository verification

**Files:**
- Modify: `README.md`
- Modify: `src/helix/cli.py`
- Test: repository verification commands

- [ ] **Step 1: Update help and README text to make CLI options primary**

```markdown
- Prefer `--npu-devices` and `--workers-per-npu` for new invocations.
- Legacy `HELIX_BATCH_NPU_DEVICES` and
  `HELIX_BATCH_WORKERS_PER_NPU` remain supported as compatibility fallback.
- Explicit CLI options override legacy environment variables.
- Managed run-eval MCP continues to ignore workers-per-npu for runtime device leasing.
```

- [ ] **Step 2: Run the focused documentation/help tests**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_cli.py -k "help or environment"
```

Expected: PASS.

- [ ] **Step 3: Run repository verification**

Run:

```bash
uv run --group dev ruff check
```

Expected: PASS.

Run:

```bash
uv run pyright
```

Expected: PASS.

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md src/helix/cli.py
git commit -m "docs: describe batch affinity cli options"
```
