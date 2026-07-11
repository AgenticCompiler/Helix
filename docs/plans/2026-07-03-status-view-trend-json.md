# Status View Trend And JSON Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--view best|trend` and `--format json` to `helix status`.

**Architecture:** Keep status artifact parsing in `src/helix/status/core.py`, expose all comparable round speedups on `OptimizeStatusWorkspace`, and keep output-specific logic in `src/helix/status/render.py`. The command handler stays thin: resolve input, inspect workspaces, then pass `view` and `format` to the renderer.

**Tech Stack:** Python 3.12, `argparse`, dataclasses, `json`, existing `unittest` tests.

---

### Task 1: Add CLI Parser Coverage

**Files:**
- Modify: `tests/test_cli.py`
- Later modify: `src/helix/cli.py`

- [ ] **Step 1: Write failing parser tests**

Add tests near the existing status parser tests:

```python
def test_status_accepts_view_option(self) -> None:
    parser = build_parser()

    args = parser.parse_args(["status", "-i", "kernels", "--view", "trend"])

    self.assertEqual(args.command_kind, CommandKind.STATUS)
    self.assertEqual(args.view, "trend")

def test_status_defaults_to_best_view(self) -> None:
    parser = build_parser()

    args = parser.parse_args(["status", "-i", "kernels"])

    self.assertEqual(args.view, "best")

def test_status_accepts_json_format(self) -> None:
    parser = build_parser()

    args = parser.parse_args(["status", "-i", "kernels", "--format", "json"])

    self.assertEqual(args.format, "json")
```

- [ ] **Step 2: Run parser tests and confirm failure**

Run:

```bash
uv run python -m unittest tests.test_cli.CliParserTests -v
```

Expected: FAIL because `--view` is unknown and `json` is not an accepted format.

- [ ] **Step 3: Implement parser support**

In `src/helix/cli.py`:

```python
_FORMAT_CHOICES = ("text", "markdown", "json")
_STATUS_VIEW_CHOICES = ("best", "trend")
```

Add a status-specific parser argument where command specs are expanded:

```python
if command_kind == CommandKind.STATUS:
    subparser.add_argument("--view", default="best", choices=_STATUS_VIEW_CHOICES)
```

- [ ] **Step 4: Re-run parser tests**

Run:

```bash
uv run python -m unittest tests.test_cli.CliParserTests -v
```

Expected: PASS.

### Task 2: Preserve Per-Round Speedups In Status Core

**Files:**
- Modify: `src/helix/optimize/models.py`
- Modify: `src/helix/status/core.py`
- Modify: `tests/test_status.py`

- [ ] **Step 1: Write failing core test**

Add a test using two comparable rounds:

```python
def test_inspect_optimize_status_workspace_returns_round_speedup_trend(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
        (workspace / "kernel_perf.txt").write_text(
            "latency-a: 10\nlatency-b: 20\n",
            encoding="utf-8",
        )
        round_one = workspace / "opt-round-1"
        round_two = workspace / "opt-round-2"
        round_one.mkdir()
        round_two.mkdir()
        (round_one / "opt_kernel_perf.txt").write_text(
            "latency-a: 8\nlatency-b: 16\n",
            encoding="utf-8",
        )
        (round_two / "opt_kernel_perf.txt").write_text(
            "latency-a: 5\nlatency-b: 10\n",
            encoding="utf-8",
        )

        status = inspect_optimize_status_workspace(workspace)

        self.assertEqual([round.round_name for round in status.rounds], ["round-1", "round-2"])
        self.assertAlmostEqual(status.rounds[0].geomean_speedup, (10 / 8 * 20 / 16) ** 0.5)
        self.assertAlmostEqual(status.rounds[1].geomean_speedup, (10 / 5 * 20 / 10) ** 0.5)
```

- [ ] **Step 2: Run the focused core test and confirm failure**

Run:

```bash
uv run python -m unittest tests.test_status.StatusTests.test_inspect_optimize_status_workspace_returns_round_speedup_trend -v
```

Expected: FAIL because `OptimizeStatusWorkspace` has no `rounds` field.

- [ ] **Step 3: Add the model field**

In `src/helix/optimize/models.py`:

```python
rounds: tuple[OptimizeStatusRound, ...] = ()
```

Place it at the end of `OptimizeStatusWorkspace` so existing test constructors keep working.

- [ ] **Step 4: Populate rounds in core**

In `src/helix/status/core.py`, return:

```python
rounds=tuple(comparable_rounds),
```

for both the `ok` and `warning` optimize-artifact paths. `no-session` can keep the default empty tuple.

- [ ] **Step 5: Re-run status tests**

Run:

```bash
uv run python -m unittest tests.test_status -v
```

Expected: PASS.

### Task 3: Add Best JSON And Trend Renderers

**Files:**
- Modify: `tests/test_status_render.py`
- Modify: `src/helix/status/render.py`
- Modify: `src/helix/commands/status.py`

- [ ] **Step 1: Write failing renderer tests**

Add tests covering:

```python
def test_render_optimize_status_best_json_includes_all_operators(self) -> None:
    stream = StringIO()
    results = [
        OptimizeStatusWorkspace(
            workspace=Path("/tmp/nope"),
            state="no-session",
            avg_improvement=None,
            geomean_speedup=None,
            best_round=None,
            logged_best=None,
            warnings=(),
        ),
        OptimizeStatusWorkspace(
            workspace=Path("/tmp/op"),
            state="ok",
            avg_improvement=0.25,
            geomean_speedup=1.5,
            best_round="round-2",
            logged_best="round-1",
            warnings=("numeric best round != logged best. computed speedup: 1.50x; logged speedup: 1.20x",),
            verified=True,
            verified_geomean_speedup=1.4,
        ),
    ]

    render_optimize_status_results(results, stdout=stream, output_format="json", view="best")

    payload = json.loads(stream.getvalue())
    self.assertEqual([item["name"] for item in payload["operators"]], ["nope", "op"])
    self.assertEqual(payload["operators"][1]["geomean_speedup"], 1.5)
```

```python
def test_render_optimize_status_trend_json_filters_no_session_and_fills_nulls(self) -> None:
    stream = StringIO()
    results = [
        OptimizeStatusWorkspace(
            workspace=Path("/tmp/empty"),
            state="no-session",
            avg_improvement=None,
            geomean_speedup=None,
            best_round=None,
            logged_best=None,
            warnings=(),
        ),
        OptimizeStatusWorkspace(
            workspace=Path("/tmp/op"),
            state="ok",
            avg_improvement=0.25,
            geomean_speedup=1.5,
            best_round="round-2",
            logged_best=None,
            warnings=(),
            rounds=(
                OptimizeStatusRound("round-1", "auto", 0.1, 1.1, 9.0),
                OptimizeStatusRound("round-3", "auto", 0.3, 1.3, 7.0),
            ),
        ),
    ]

    render_optimize_status_results(results, stdout=stream, output_format="json", view="trend")

    payload = json.loads(stream.getvalue())
    self.assertEqual(payload["operators"], [
        {
            "name": "op",
            "round_speedups": {
                "round-1": 1.1,
                "round-3": 1.3,
            },
        }
    ])
```

Also add text and Markdown trend table tests asserting `Name`, `round-1`, `round-2`, sorted operators, and `-` for missing speedups.

- [ ] **Step 2: Run render tests and confirm failure**

Run:

```bash
uv run python -m unittest tests.test_status_render -v
```

Expected: FAIL because `view` and JSON/trend rendering do not exist.

- [ ] **Step 3: Implement rendering**

In `src/helix/status/render.py`:

- add `import json`
- update `render_optimize_status_results(..., view: str = "best")`
- route `view == "best"` to existing text/Markdown behavior plus new JSON
- route `view == "trend"` to new text/Markdown/JSON helpers
- build trend rounds from non-`no-session` results:

```python
round_names = sorted(
    {round.round_name for item in rows for round in item.rounds},
    key=_round_sort_key,
)
```

- [ ] **Step 4: Pass view from the command handler**

In `src/helix/commands/status.py`, pass:

```python
view=str(getattr(args, "view", "best")),
```

to `render_optimize_status_results()` in both single-workspace and batch paths.

- [ ] **Step 5: Re-run render tests**

Run:

```bash
uv run python -m unittest tests.test_status_render -v
```

Expected: PASS.

### Task 4: CLI Integration, Docs, And Verification

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `src/helix/cli.py`
- Modify: `src/helix/commands/status.py`
- Modify: `src/helix/status/core.py`
- Modify: `src/helix/status/render.py`
- Modify: `src/helix/optimize/models.py`

- [ ] **Step 1: Add CLI integration tests**

Add focused `main()` tests for:

- `status --format json --view best`
- `status --format json --view trend`
- `status --view trend --format markdown`

Use temporary workspaces with baseline perf and `opt-round-*` perf artifacts.

- [ ] **Step 2: Run CLI integration tests and confirm failure if implementation is incomplete**

Run:

```bash
uv run python -m unittest tests.test_cli -v
```

Expected: PASS after Tasks 1-3 are complete.

- [ ] **Step 3: Update README**

In the status section, document:

```bash
uv run helix status --input operators_root --view trend
uv run helix status --input operators_root --view trend --format json
uv run helix status --input operators_root --format json
```

Explain that `--view best` is default, `--view trend` emits per-round geomean
speedups, and JSON uses top-level `operators` with raw float speedups.

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run python -m unittest tests.test_status tests.test_status_render tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Run standard repository verification**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS.
