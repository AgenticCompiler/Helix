# Optimize Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optimize-workspace upload support with a dedicated `upload-optimize` command, automatic post-optimize upload controlled by `--no-upload`, and a same-repo standalone upload server project under `services/triton-agent-upload-server/`.

**Architecture:** Keep all client logic inside the main repository’s CLI tree through a new `src/triton_agent/optimize_upload/` feature package and reuse that shared workflow from both `upload-optimize` and `optimize` auto-upload. Implement the HTTP receiver as an isolated service project under `services/triton-agent-upload-server/` with its own `pyproject.toml`, dependencies, tests, and runtime entrypoint so deployment stays separate while Git history remains unified.

**Tech Stack:** Python 3, `argparse`, `unittest`, `tarfile`, `urllib`, FastAPI, uvicorn, repository `uv` workflows

---

## File Map

- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/output.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_commands.py`
- Modify: `tests/test_optimize_runtime.py`
- Create: `src/triton_agent/commands/upload_optimize.py`
- Create: `src/triton_agent/optimize_upload/__init__.py`
- Create: `src/triton_agent/optimize_upload/models.py`
- Create: `src/triton_agent/optimize_upload/naming.py`
- Create: `src/triton_agent/optimize_upload/collector.py`
- Create: `src/triton_agent/optimize_upload/manifest.py`
- Create: `src/triton_agent/optimize_upload/packager.py`
- Create: `src/triton_agent/optimize_upload/client.py`
- Create: `src/triton_agent/optimize_upload/workflow.py`
- Create: `tests/test_optimize_upload.py`
- Create: `services/triton-agent-upload-server/pyproject.toml`
- Create: `services/triton-agent-upload-server/README.md`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/__init__.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/app.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/config.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/routes.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/models.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/naming.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/storage.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/responses.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/auth.py`
- Create: `services/triton-agent-upload-server/tests/test_healthz.py`
- Create: `services/triton-agent-upload-server/tests/test_upload_route.py`
- Create: `services/triton-agent-upload-server/tests/test_storage.py`
- Create: `services/triton-agent-upload-server/tests/test_naming.py`

## Task 1: Lock In CLI Surface And Option Plumbing

**Files:**
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_commands.py`

- [ ] **Step 1: Write failing parser tests for `upload-optimize` and `--no-upload`**

Add CLI coverage in `tests/test_cli.py` for:

```python
    def test_upload_optimize_command_parses_input_and_verbose(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["upload-optimize", "-i", "workspace-root", "--verbose"]
        )
        self.assertEqual(args.command, "upload-optimize")
        self.assertEqual(args.input, "workspace-root")
        self.assertTrue(args.verbose)

    def test_optimize_command_defaults_upload_enabled(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py"])
        options = optimize_run_options_from_args(args)
        self.assertTrue(options.upload_enabled)

    def test_optimize_command_accepts_no_upload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["optimize", "-i", "kernel.py", "--no-upload"])
        options = optimize_run_options_from_args(args)
        self.assertFalse(options.upload_enabled)

    def test_optimize_batch_accepts_no_upload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["optimize-batch", "-i", "workspace-root", "--no-upload"]
        )
        options = optimize_run_options_from_args(args)
        self.assertFalse(options.upload_enabled)
```

- [ ] **Step 2: Run the parser tests to confirm they fail**

Run:
`uv run python -m unittest tests.test_cli.CliParserTests.test_upload_optimize_command_parses_input_and_verbose tests.test_cli.CliParserTests.test_optimize_command_defaults_upload_enabled tests.test_cli.CliParserTests.test_optimize_command_accepts_no_upload tests.test_cli.CliParserTests.test_optimize_batch_accepts_no_upload -v`

Expected: `FAIL` because the new subcommand and option are not yet wired.

- [ ] **Step 3: Implement parser wiring and optimize option plumbing**

Make these concrete changes:

- in `src/triton_agent/optimize/models.py`, extend `OptimizeRunOptions` with:

```python
    upload_enabled: bool = True
```

- in `src/triton_agent/commands/optimize.py`, map:

```python
        upload_enabled=not bool(getattr(args, "no_upload", False)),
```

- in `src/triton_agent/cli.py`:
  - add a new command spec entry for `upload-optimize`
  - expose `--verbose`
  - reuse `-i/--input`
  - add `--no-upload` only on `optimize` and `optimize-batch`
  - add `TRITON_AGENT_OPTIMIZE_UPLOAD_URL` to help text

- [ ] **Step 4: Add a thin command handler stub for `upload-optimize`**

Create `src/triton_agent/commands/upload_optimize.py` with a temporary handler that validates the input path exists and raises `NotImplementedError` or a temporary `ValueError` with a clear message. This keeps parser wiring compilable before full implementation.

- [ ] **Step 5: Re-run the parser tests and make them pass**

Run:
`uv run python -m unittest tests.test_cli.CliParserTests.test_upload_optimize_command_parses_input_and_verbose tests.test_cli.CliParserTests.test_optimize_command_defaults_upload_enabled tests.test_cli.CliParserTests.test_optimize_command_accepts_no_upload tests.test_cli.CliParserTests.test_optimize_batch_accepts_no_upload -v`

Expected: `PASS`

- [ ] **Step 6: Commit the CLI surface changes**

```bash
git add src/triton_agent/cli.py src/triton_agent/commands/optimize.py src/triton_agent/commands/upload_optimize.py src/triton_agent/optimize/models.py tests/test_cli.py tests/test_optimize_commands.py README.md
git commit -m "feat: add optimize upload cli surface"
```

## Task 2: Build The Client Naming, Manifest, And Collection Core

**Files:**
- Create: `src/triton_agent/optimize_upload/__init__.py`
- Create: `src/triton_agent/optimize_upload/models.py`
- Create: `src/triton_agent/optimize_upload/naming.py`
- Create: `src/triton_agent/optimize_upload/collector.py`
- Create: `src/triton_agent/optimize_upload/manifest.py`
- Create: `tests/test_optimize_upload.py`

- [ ] **Step 1: Write failing tests for workspace slugging and upload identity**

Add `tests/test_optimize_upload.py` with:

```python
class OptimizeUploadNamingTests(unittest.TestCase):
    def test_slugify_workspace_name_preserves_safe_characters(self) -> None:
        self.assertEqual(slugify_workspace_name("matmul_case-01"), "matmul_case-01")

    def test_slugify_workspace_name_replaces_unsafe_characters(self) -> None:
        self.assertEqual(slugify_workspace_name("matmul case/01"), "matmul_case_01")

    def test_slugify_workspace_name_falls_back_to_workspace(self) -> None:
        self.assertEqual(slugify_workspace_name("////"), "workspace")

    def test_build_upload_identity_uses_workspace_name_and_slug(self) -> None:
        identity = build_upload_identity(Path("/tmp/matmul case"))
        self.assertEqual(identity.workspace_name, "matmul case")
        self.assertEqual(identity.workspace_slug, "matmul_case")
        self.assertRegex(identity.upload_uid, r"^[0-9a-f]{32}$")
        self.assertRegex(identity.upload_timestamp, r"^\d{8}T\d{6}Z$")
```

- [ ] **Step 2: Write failing tests for whitelist collection**

Add focused collection tests that build a temp workspace containing:

- root files:
  - `kernel.py`
  - `opt_kernel.py`
  - `test_kernel.py`
  - `differential_test_kernel.py`
  - `bench_kernel.py`
  - `opt-note.md`
  - `learned_lessons.md`
- `baseline/state.json` that points to a prefixed perf file
- `baseline/baseline_kernel.py`
- `baseline/kernel_perf.txt`
- `opt-round-1/summary.md`
- `opt-round-1/attempts.md`
- `opt-round-1/round-state.json`
- `opt-round-1/opt_kernel.py`
- `opt-round-1/opt_kernel_perf.txt`
- `opt-round-1/perf-analysis.md`
- `opt-round-1/compiler-analysis.md`
- `triton-agent-logs/optimize.show-output.log`
- excluded paths:
  - `opt-round-1/ir/`
  - `opt-verify/verify-1/verify-state.json`
  - `foo_result.pt`
  - `PROF_123/`
  - `baseline/archive.tar.gz`

Assert that collection includes only the whitelist and excludes the forbidden entries.

- [ ] **Step 3: Run the new tests to confirm they fail**

Run:
`uv run python -m unittest tests.test_optimize_upload.OptimizeUploadNamingTests tests.test_optimize_upload.OptimizeUploadCollectorTests -v`

Expected: `FAIL` because the new upload package does not exist yet.

- [ ] **Step 4: Implement upload models and naming helpers**

In `src/triton_agent/optimize_upload/models.py`, add focused dataclasses such as:

```python
@dataclass(frozen=True)
class UploadIdentity:
    upload_uid: str
    upload_timestamp: str
    workspace_name: str
    workspace_slug: str


@dataclass(frozen=True)
class CollectedUpload:
    workspace: Path
    operator_file: Path | None
    included_files: tuple[Path, ...]
    excluded_entries: tuple[tuple[str, str], ...]
```

In `src/triton_agent/optimize_upload/naming.py`, implement:

- `slugify_workspace_name(name: str) -> str`
- `build_upload_identity(workspace: Path) -> UploadIdentity`

Use `re.sub` for slug normalization and `datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")` for timestamps.

- [ ] **Step 5: Implement whitelist collection**

In `src/triton_agent/optimize_upload/collector.py`:

- validate the workspace path
- find root-level whitelist files
- resolve the baseline perf artifact and baseline operator from `baseline/state.json`
- inspect each `opt-round-*` directory and collect only:
  - round operator `.py`
  - `attempts.md`
  - `summary.md`
  - `round-state.json`
  - one round perf artifact
  - `perf-analysis.md`
  - `compiler-analysis.md`
- collect `*.show-output.log` under `triton-agent-logs/`
- record excluded entries only for the explicitly forbidden known paths you discover during traversal

Do not import the upload server code here.

- [ ] **Step 6: Implement manifest generation**

In `src/triton_agent/optimize_upload/manifest.py`, add a function:

```python
def build_manifest(
    identity: UploadIdentity,
    collected: CollectedUpload,
) -> dict[str, object]:
    ...
```

The payload must include:

- `manifest_version`
- `upload_uid`
- `upload_timestamp`
- `workspace_name`
- `workspace_slug`
- `operator_file`
- `included_files`
- `excluded_entries`
- `file_count`
- `total_bytes`

- [ ] **Step 7: Re-run the focused tests and make them pass**

Run:
`uv run python -m unittest tests.test_optimize_upload.OptimizeUploadNamingTests tests.test_optimize_upload.OptimizeUploadCollectorTests -v`

Expected: `PASS`

- [ ] **Step 8: Commit the collection and manifest core**

```bash
git add src/triton_agent/optimize_upload/__init__.py src/triton_agent/optimize_upload/models.py src/triton_agent/optimize_upload/naming.py src/triton_agent/optimize_upload/collector.py src/triton_agent/optimize_upload/manifest.py tests/test_optimize_upload.py
git commit -m "feat: add optimize upload collection core"
```

## Task 3: Implement Packaging And HTTP Upload Workflow

**Files:**
- Create: `src/triton_agent/optimize_upload/packager.py`
- Create: `src/triton_agent/optimize_upload/client.py`
- Create: `src/triton_agent/optimize_upload/workflow.py`
- Create: `tests/test_optimize_upload.py`

- [ ] **Step 1: Write failing tests for tarball layout**

Extend `tests/test_optimize_upload.py` with a packager test that:

- builds a collected upload from a temp workspace
- writes a tarball
- opens it with `tarfile`
- asserts members include:
  - `baseline/state.json`
  - `opt-round-1/summary.md`
  - `triton-agent-logs/optimize.show-output.log`
  - `_upload/manifest.json`
- asserts no extra top-level wrapper directory is present

- [ ] **Step 2: Write failing tests for HTTP request construction**

Add a fake HTTP server or patch `urllib.request.urlopen` to assert:

- request method is `POST`
- `Content-Type` is `application/gzip`
- `Content-Length` is present
- all `X-Triton-Agent-*` headers are present
- request body bytes match the tarball file

Add failure tests for:

- missing `TRITON_AGENT_OPTIMIZE_UPLOAD_URL`
- malformed JSON response
- HTTP error surface

- [ ] **Step 3: Run the workflow tests to confirm they fail**

Run:
`uv run python -m unittest tests.test_optimize_upload.OptimizeUploadPackagerTests tests.test_optimize_upload.OptimizeUploadClientTests -v`

Expected: `FAIL`

- [ ] **Step 4: Implement tarball creation**

In `src/triton_agent/optimize_upload/packager.py`, implement a helper such as:

```python
@contextmanager
def build_upload_tarball(
    collected: CollectedUpload,
    manifest: dict[str, object],
) -> Iterator[Path]:
    ...
```

Requirements:

- create a temp `.tar.gz`
- add collected files using workspace-relative archive names
- write the manifest as `_upload/manifest.json`
- clean up the temp tarball in a `finally` block

- [ ] **Step 5: Implement HTTP upload client**

In `src/triton_agent/optimize_upload/client.py`, implement:

- `load_upload_url() -> str`
- `upload_tarball(identity: UploadIdentity, tarball: Path, url: str) -> UploadResponse`

Use `urllib.request.Request` plus `urlopen`. Parse JSON into a small dataclass in `models.py`.

- [ ] **Step 6: Implement shared upload workflow**

In `src/triton_agent/optimize_upload/workflow.py`, add a high-level function such as:

```python
def upload_optimize_workspace(
    workspace: Path,
    *,
    verbose: bool = False,
) -> UploadResponse:
    ...
```

This function should:

- validate workspace
- build identity
- collect files
- build manifest
- build temp tarball
- load the upload URL
- upload the tarball

- [ ] **Step 7: Re-run the focused upload workflow tests**

Run:
`uv run python -m unittest tests.test_optimize_upload.OptimizeUploadPackagerTests tests.test_optimize_upload.OptimizeUploadClientTests tests.test_optimize_upload.OptimizeUploadWorkflowTests -v`

Expected: `PASS`

- [ ] **Step 8: Commit packaging and client upload**

```bash
git add src/triton_agent/optimize_upload/packager.py src/triton_agent/optimize_upload/client.py src/triton_agent/optimize_upload/workflow.py tests/test_optimize_upload.py
git commit -m "feat: add optimize upload workflow"
```

## Task 4: Integrate `upload-optimize` And Auto-Upload Behavior

**Files:**
- Modify: `src/triton_agent/commands/upload_optimize.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `tests/test_optimize_commands.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Write failing tests for `upload-optimize` handler**

In `tests/test_optimize_commands.py`, add tests that:

- patch `upload_optimize_workspace`
- assert `handle_upload_optimize` calls it with the resolved workspace path
- assert success returns `0`
- assert missing URL surfaces a non-zero exit path

- [ ] **Step 2: Write failing tests for optimize auto-upload behavior**

In `tests/test_optimize_runtime.py`, add tests that:

- successful optimize with `upload_enabled=True` calls the upload workflow
- successful optimize with `upload_enabled=False` does not call it
- failed optimize does not call it
- missing URL during auto-upload becomes a skipped behavior
- upload failure during auto-upload preserves the optimize exit code

- [ ] **Step 3: Run the integration tests to confirm they fail**

Run:
`uv run python -m unittest tests.test_optimize_commands tests.test_optimize_runtime -v`

Expected: `FAIL`

- [ ] **Step 4: Implement the `upload-optimize` command handler**

In `src/triton_agent/commands/upload_optimize.py`, replace the stub with real logic:

- resolve and validate `--input`
- call `upload_optimize_workspace`
- print one short summary on success
- print one short error summary on failure
- return non-zero on failure

- [ ] **Step 5: Integrate auto-upload into optimize flows**

In `src/triton_agent/commands/optimize.py`:

- after a successful single-workspace `run_optimize_request`, call the upload workflow when `options.upload_enabled`
- on missing URL, treat the upload as skipped
- on upload failure, preserve `result.return_code`
- only print upload success/failure/skipped lines when `request.verbose`

In `src/triton_agent/optimize/batch.py`, integrate the same shared workflow per successful workspace when batch options allow upload.

- [ ] **Step 6: Re-run the integration tests and make them pass**

Run:
`uv run python -m unittest tests.test_optimize_commands tests.test_optimize_runtime -v`

Expected: `PASS`

- [ ] **Step 7: Commit CLI integration**

```bash
git add src/triton_agent/commands/upload_optimize.py src/triton_agent/commands/optimize.py src/triton_agent/optimize/batch.py tests/test_optimize_commands.py tests/test_optimize_runtime.py
git commit -m "feat: integrate optimize upload flows"
```

## Task 5: Create The Standalone Upload Server Project Under `services/`

**Files:**
- Create: `services/triton-agent-upload-server/pyproject.toml`
- Create: `services/triton-agent-upload-server/README.md`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/__init__.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/app.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/config.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/routes.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/models.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/naming.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/storage.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/responses.py`
- Create: `services/triton-agent-upload-server/src/triton_agent_upload_server/auth.py`
- Create: `services/triton-agent-upload-server/tests/test_healthz.py`
- Create: `services/triton-agent-upload-server/tests/test_upload_route.py`
- Create: `services/triton-agent-upload-server/tests/test_storage.py`
- Create: `services/triton-agent-upload-server/tests/test_naming.py`

- [ ] **Step 1: Write the failing server naming and storage tests**

Add tests in `services/triton-agent-upload-server/tests/test_naming.py` for:

- safe slug normalization
- archive filename construction:
  - `<timestamp>-<workspace_slug>-<uid>.tar.gz`
- receipt filename construction:
  - `<timestamp>-<workspace_slug>-<uid>.receipt.json`

Add tests in `test_storage.py` for:

- temp file write then atomic publish
- duplicate target rejection
- sidecar receipt creation

- [ ] **Step 2: Write the failing route tests**

Add FastAPI test-client coverage in `test_upload_route.py` for:

- `GET /healthz`
- valid `POST /uploads`
- missing headers -> `400`
- missing `Content-Length` -> `411`
- wrong `Content-Type` -> `415`
- oversized upload -> `413`
- duplicate upload -> `409`

- [ ] **Step 3: Run the server tests to confirm they fail**

Run:
`cd services/triton-agent-upload-server && uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: `FAIL`

- [ ] **Step 4: Create the standalone service project metadata**

Add `services/triton-agent-upload-server/pyproject.toml` with:

- Python `>=3.11`
- dependencies:
  - `fastapi`
  - `uvicorn`
  - `pytest`
  - `httpx`

Expose a console script:

```toml
[project.scripts]
triton-agent-upload-server = "triton_agent_upload_server.app:main"
```

- [ ] **Step 5: Implement naming, config, models, and storage helpers**

Implement:

- `naming.py`
  - slug normalization
  - header validation
  - filename builders
- `config.py`
  - app settings
  - size limits
  - storage root / temp root
- `models.py`
  - response and receipt models
- `storage.py`
  - stream body to temp file
  - verify byte count
  - atomically publish archive
  - write receipt file

- [ ] **Step 6: Implement FastAPI app and routes**

In `app.py`, provide:

- `create_app()`
- `main()` that launches uvicorn

In `routes.py`, provide:

- `GET /healthz`
- `POST /uploads`

The upload route must:

- validate headers and content type
- call a placeholder `authorize_request` hook from `auth.py`
- stream the body to storage
- return the JSON success payload

- [ ] **Step 7: Re-run the server test suite**

Run:
`cd services/triton-agent-upload-server && uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

Expected: `PASS`

- [ ] **Step 8: Commit the server project**

```bash
git add services/triton-agent-upload-server
git commit -m "feat: add optimize upload server project"
```

## Task 6: Update Docs And Verify End-To-End Expectations

**Files:**
- Modify: `README.md`
- Modify: `docs/specs/2026-05-26-optimize-upload-design.md`
- Create: `docs/plans/2026-05-26-optimize-upload-implementation-plan.md`

- [ ] **Step 1: Update README client documentation**

Add:

- `upload-optimize`
- `--no-upload`
- `TRITON_AGENT_OPTIMIZE_UPLOAD_URL`
- note that the upload server project lives under `services/triton-agent-upload-server/`

- [ ] **Step 2: Update README server development notes**

Document:

- where the server project lives
- how to start it locally
- where archives are stored
- that the server stores raw `.tar.gz` plus `.receipt.json`

- [ ] **Step 3: Run targeted repository tests**

Run:
`uv run python -m unittest tests.test_cli tests.test_optimize_commands tests.test_optimize_runtime tests.test_optimize_upload -v`

Expected: `PASS`

- [ ] **Step 4: Run repository verification commands**

Run:
`uv run --group dev ruff check`

Run:
`uv run pyright`

Run:
`uv run python -m unittest discover -s tests -v`

Expected: all pass, or record any unrelated pre-existing failures before merging.

- [ ] **Step 5: Final commit**

```bash
git add README.md docs/specs/2026-05-26-optimize-upload-design.md docs/plans/2026-05-26-optimize-upload-implementation-plan.md
git commit -m "docs: add optimize upload implementation plan"
```
