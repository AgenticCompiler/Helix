# Optimize Upload Design

## Summary

- Add an `upload-optimize` subcommand that packages one optimize workspace and uploads it to a configured HTTP endpoint.
- Add `--no-upload` to `optimize` and `optimize-batch`, with automatic upload enabled by default.
- Add a feature-local client package under `src/helix/optimize_upload/` for workspace collection, manifest generation, tarball creation, and HTTP upload.
- Upload only an explicit optimize-artifact whitelist needed for later analysis.
- Configure the upload destination with `HELIX_OPTIMIZE_UPLOAD_URL`.
- Keep the upload server out of the main `helix` Python package. It should live as a separately deployable service project under the repository `services/` tree and save the original uploaded `.tar.gz` package plus a small sidecar receipt.

## Goals

- Make it easy to hand off one completed optimize workspace for later offline analysis.
- Keep optimize upload behavior explicit, inspectable, and easy to debug through a dedicated CLI subcommand.
- Reuse one shared Python upload workflow from both `upload-optimize` and automatic post-optimize upload.
- Preserve current optimize success semantics: upload failures must not turn a successful optimize run into a failed CLI exit.
- Keep upload payloads focused on operator code, test and benchmark harnesses, optimize summaries, and perf artifacts.
- Avoid uploading bulky profiler directories, PT result files, IR trees, or other raw capture payloads that are not needed for the intended analysis flow.

## Non-Goals

- Do not add the upload server implementation to `src/helix/` or mix it into the main CLI package.
- Do not make the server unpack or validate optimize-specific contents before saving uploads.
- Do not add authentication in the first version.
- Do not upload entire workspaces through recursive directory packaging plus a blacklist.
- Do not upload `ir/`, `opt-verify/`, `agent-sessions.jsonl`, profiler output trees, or PT artifacts.
- Do not change optimize round artifact formats in this design.
- Do not add retry orchestration, resumable upload, deduplication, or chunked upload protocols in this version.

## User-Facing Behavior

### New Subcommand

Add a dedicated command:

```bash
uv run helix upload-optimize --input /path/to/workspace
```

Behavior:

- `--input` must point to one optimize workspace root directory.
- The command validates the workspace, collects the upload whitelist, builds a temporary `.tar.gz`, uploads it, and prints a short success or failure summary.
- The command returns non-zero when validation, packaging, configuration, upload, or response parsing fails.
- `--verbose` may print extra detail such as included file count, filtered file count, tarball size, and the server response payload.

### Automatic Upload From Optimize

Add a new option on both `optimize` and `optimize-batch`:

```bash
--no-upload
```

Behavior:

- Automatic upload is enabled by default.
- `--no-upload` disables automatic upload entirely.
- Automatic upload is attempted only after the optimize run itself succeeds.
- In `optimize-batch`, upload is attempted only for each workspace whose optimize run succeeds.
- Automatic upload failures do not change the optimize command exit code.
- Automatic upload is silent by default.
- When `--verbose` is enabled:
  - success prints a short upload summary
  - failure prints a short warning summary
  - missing configuration prints a short skipped summary

### Upload URL Configuration

Add one environment variable:

```text
HELIX_OPTIMIZE_UPLOAD_URL
```

Behavior:

- The value is a complete HTTP upload endpoint, for example:

```text
http://10.0.0.8:8080/uploads
```

- `upload-optimize` fails when the variable is unset.
- Automatic upload from `optimize` and `optimize-batch` skips the upload when the variable is unset.
- That skip remains silent unless `--verbose` is enabled.

## Upload Scope

Uploads use an explicit whitelist. The client must not package the entire workspace and then filter it opportunistically.

### Included Workspace-Root Files

Include these workspace-root files when present:

- the source operator `.py`
- `opt_*.py`
- `test_*.py`
- `differential_test_*.py`
- `bench_*.py`
- `opt-note.md`
- `learned_lessons.md`

The source operator should come from the workspace’s optimize context rather than from an arbitrary `.py` glob when that context is available.

### Included `baseline/`

Include:

- `baseline/state.json`
- the baseline operator snapshot referenced by `baseline/state.json`
- the baseline perf artifact referenced by `baseline/state.json`

Do not hardcode baseline perf upload to `baseline/perf.txt`; use the path recorded in baseline state so prefixed perf filenames continue to work.

### Included `opt-round-*/`

For each round directory, include:

- the round-local optimized operator `.py`
- `attempts.md`
- `summary.md`
- `round-state.json`
- the round perf artifact for that round
- `perf-analysis.md`
- `compiler-analysis.md`

Round perf artifact resolution should follow the round contract and current optimize artifact rules rather than assume a fixed filename.

### Included `helix-logs/`

Include only:

- `*.show-output.log`

No other log archive files are uploaded in this design.

### Excluded Content

Explicitly exclude:

- `ir/`
- `opt-verify/`
- `agent-sessions.jsonl`
- any `.pt` file
- any `PROF_*` directory
- any `ASCEND_PROFILER_OUTPUT/` directory
- any `mindstudio_profiler_output/` directory
- any `extra-info/` directory
- cache directories such as `__pycache__/`
- virtual environments and agent staging directories
- existing archive files such as `.tar`, `.tar.gz`, `.tgz`, or `.zip`

## Workspace Validation

`upload-optimize` should fail fast with a short actionable error when the input directory does not look like an optimize workspace.

Minimum validity:

- the input path exists
- the input path is a directory
- the workspace contains `baseline/` or at least one `opt-round-*` directory

The upload workflow may perform deeper validation when resolving baseline or round artifacts, but it should not require a fully successful optimize-status rendering pipeline just to begin packaging.

## Package Format

The client creates one temporary gzip-compressed tarball.

The tarball should contain:

- only whitelist-selected files
- one manifest file at `_upload/manifest.json`

The tarball should not wrap the entire workspace in an extra top-level directory. Archive members should be rooted at the workspace-relative paths, for example:

```text
baseline/state.json
opt-round-1/summary.md
helix-logs/optimize.show-output.log
_upload/manifest.json
```

This keeps the archive straightforward for later offline unpacking and scripting.

## Manifest Contract

The client-generated `_upload/manifest.json` should be the package’s authoritative metadata.

Include at least:

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

`excluded_entries` should be compact and reason-oriented rather than a full recursive inventory of every skipped file under the workspace.

The manifest exists for later offline analysis after the archive is saved by the server. The server does not need to parse it in the first version.

## Upload Identity

The client should generate three identity values for each upload:

- `upload_uid`
- `upload_timestamp`
- `workspace_name`

Recommended rules:

- `upload_uid`
  - generate with `uuid.uuid4().hex`
  - store and transmit as 32 lowercase hex characters
- `upload_timestamp`
  - generate in UTC
  - format as `YYYYMMDDTHHMMSSZ`
  - example: `20260526T141530Z`
- `workspace_name`
  - use the input workspace directory name

The client should derive `workspace_slug` from `workspace_name` with the same normalization rules expected by the external server.

## Upload Protocol

The client uploads the package through one HTTP request:

- method: `POST`
- URL: `HELIX_OPTIMIZE_UPLOAD_URL`
- body: raw `.tar.gz` bytes
- content type: `application/gzip`

The client should send `Content-Length` based on the tarball size so the server can enforce upload size limits without buffering the whole request in memory.

Add these request headers:

- `X-Helix-Upload-Uid`
- `X-Helix-Upload-Timestamp`
- `X-Helix-Workspace-Name`
- `X-Helix-Workspace-Slug`
- `X-Helix-Manifest-Version`

These headers let a thin external server name the saved archive without unpacking it.

The first-version client does not send auth tokens, signatures, or custom retry metadata.

## External Server Contract

The upload server is a separate deployable program inside this repository, but outside the main CLI package and runtime tree.

### Recommended Repository

Implement the server as its own service project under:

```text
services/helix-upload-server/
```

Recommended rationale:

- keep Git management unified in one repository
- keep deployment and runtime boundaries separate from the main CLI package
- keep server-side auth, storage, and operations concerns out of `src/helix/`
- allow the service to keep its own dependency set and launch commands without complicating the root package

### Recommended Runtime Stack

Recommended stack:

- Python `3.11+`
- `uv` for environment and dependency management
- `FastAPI` for HTTP routing and JSON responses
- `uvicorn` as the ASGI server

Recommended rationale:

- the service is small, but likely to grow auth and operational concerns later
- FastAPI gives straightforward request handling, JSON responses, and future middleware hooks
- the service can still keep its implementation minimal while avoiding raw `http.server` plumbing

An all-stdlib server would work for a one-endpoint prototype, but it is not the recommended design target for the standalone service.

### Recommended Server Project Layout

Suggested repository structure:

```text
services/
  helix-upload-server/
    pyproject.toml
    README.md
    src/
      helix_upload_server/
        __init__.py
        app.py
        config.py
        routes.py
        models.py
        naming.py
        storage.py
        responses.py
        auth.py
    tests/
      test_healthz.py
      test_upload_route.py
      test_storage.py
      test_naming.py
```

Suggested responsibilities:

- `app.py`
  - FastAPI app factory
  - startup wiring
- `config.py`
  - CLI flags or environment configuration
  - storage root
  - temp root
  - size limits
  - log level
- `routes.py`
  - `GET /healthz`
  - `POST /uploads`
- `models.py`
  - response payload models
  - receipt model
- `naming.py`
  - workspace slug normalization
  - filename construction
  - header validation helpers
- `storage.py`
  - stream request body into temp files
  - reserve final target names
  - atomically publish final `.tar.gz`
  - write sidecar receipt
- `responses.py`
  - consistent JSON error payloads
- `auth.py`
  - placeholder authorization hook for future token-based auth

### Recommended Startup Interface

The standalone server should expose a console entrypoint such as:

```bash
cd /path/to/helix/services/helix-upload-server
uv run helix-upload-server \
  --host 0.0.0.0 \
  --port 8080 \
  --storage-root /data/helix/uploads \
  --temp-root /data/helix/uploads/.tmp \
  --max-upload-bytes 536870912
```

Recommended startup options:

- `--host`
- `--port`
- `--storage-root`
- `--temp-root`
- `--max-upload-bytes`
- `--log-level`

Equivalent environment variables are optional, but the CLI flags above should be sufficient for the first implementation.

### Responsibilities

The server should:

- expose a health endpoint such as `GET /healthz`
- accept `POST /uploads`
- validate required headers and content type
- validate `Content-Length`
- stream the request body to a temporary file
- atomically save the final `.tar.gz`
- write a small sidecar receipt file
- return a JSON response describing the stored upload

### Request Validation

Recommended request validation rules:

- require `Content-Type: application/gzip`
- require `Content-Length`
- reject requests whose `Content-Length` exceeds `--max-upload-bytes`
- require all custom upload headers
- require `upload_uid` to match lowercase 32-hex format
- require `upload_timestamp` to match `YYYYMMDDTHHMMSSZ`
- require `workspace_name` to be non-empty
- require `workspace_slug` to equal the server-side normalized slug derived from `workspace_name`

This lets the server remain thin while still rejecting malformed uploads early.

### Storage Behavior

The server must save the original uploaded archive and must not unpack it as part of normal receipt.

Store uploads as:

```text
<timestamp>-<workspace_slug>-<uid>.tar.gz
<timestamp>-<workspace_slug>-<uid>.receipt.json
```

Example:

```text
20260526T141530Z-matmul_case_01-6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12.tar.gz
20260526T141530Z-matmul_case_01-6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12.receipt.json
```

### Workspace Slug Rules

`workspace_slug` should be derived from the workspace name and normalized so it is safe in filenames:

- preserve ASCII letters, digits, `.`, `_`, and `-`
- replace other characters with `_`
- collapse repeated `_`
- fall back to `workspace` if the result would otherwise be empty

The client and server should follow the same slugging rules so response summaries remain predictable.

### Receipt File

The server-side sidecar receipt should record at least:

- `upload_uid`
- `upload_timestamp`
- `workspace_name`
- `workspace_slug`
- `received_at`
- `stored_path`
- `content_length`

The receipt may also record:

- `content_type`
- `remote_addr`
- `manifest_version`

### Suggested Response Shape

Successful responses should return JSON similar to:

```json
{
  "ok": true,
  "upload_uid": "6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12",
  "upload_timestamp": "20260526T141530Z",
  "workspace_name": "matmul_case_01",
  "workspace_slug": "matmul_case_01",
  "stored_path": "/data/helix/uploads/20260526T141530Z-matmul_case_01-6f7c2f6d9b8c4d8ab2c4f91e7f9b5a12.tar.gz"
}
```

Recommended failure codes:

- `400` for malformed headers or invalid request shape
- `411` for missing `Content-Length`
- `409` for duplicate `<timestamp>-<workspace_slug>-<uid>` targets
- `413` for uploads exceeding server limits
- `415` for unsupported content type
- `500` for unexpected server errors

### Durable Write Flow

Recommended server write algorithm:

1. parse and validate request headers
2. compute the final archive name:
   - `<timestamp>-<workspace_slug>-<uid>.tar.gz`
3. compute the final receipt name:
   - `<timestamp>-<workspace_slug>-<uid>.receipt.json`
4. create the temp root if needed
5. stream the request body into a unique temp file under the temp root
6. verify the streamed byte count matches `Content-Length`
7. fail with `409` if the final archive path or final receipt path already exists
8. publish the archive atomically into the final storage root
9. write the receipt sidecar
10. return the success JSON response

The implementation should not write directly to the final archive path while the request is still in flight. This avoids leaving partial files behind after interrupted uploads.

Recommended implementation detail:

- keep the temp root on the same filesystem as the storage root
- write to a unique `*.partial` path
- promote with an atomic rename step only after the upload completes successfully
- if a same-name target already exists, delete the temp file and return `409`

### Logging And Operations

Recommended server logs:

- request accepted
- request rejected and reason
- bytes streamed
- final stored archive path
- receipt path
- elapsed processing time

Keep logs line-oriented and machine-readable enough for later ingestion, but plain text logging is sufficient for the first version.

### Future Auth Hook

The first version does not authenticate requests, but the server should preserve a clean place to add auth later.

Recommended shape:

- `auth.py` exports a small function or dependency that currently always allows the request
- `routes.py` calls that hook before streaming the body

This keeps later token or signature support from requiring a route redesign.

### Deployment Recommendation

Recommended deployment model:

- one Linux host or VM
- persistent filesystem-backed storage root
- `systemd` service or container deployment
- reverse proxy optional, not required for the first version

If the service is placed behind a reverse proxy later, ensure the proxy’s body-size limit matches or exceeds the server’s `--max-upload-bytes`.

### Server Test Plan

Recommended server-side tests in the separate repository:

- `GET /healthz` returns success
- valid upload writes `.tar.gz` plus `.receipt.json`
- missing headers return `400`
- missing `Content-Length` returns `411`
- wrong `Content-Type` returns `415`
- oversize upload returns `413`
- duplicate target returns `409`
- temp files are removed on failure
- archive filename matches `timestamp-workspace_slug-uid.tar.gz`
- receipt payload records the expected metadata
- invalid `workspace_slug` relative to `workspace_name` is rejected
- streamed byte count mismatch is rejected
- interrupted upload does not leave a published final archive

## Client Implementation Shape

Add a new feature-local package:

```text
src/helix/optimize_upload/
```

Suggested module split:

- `models.py`
- `collector.py`
- `manifest.py`
- `packager.py`
- `client.py`
- `workflow.py`

Suggested responsibilities:

- `collector.py`
  - validate the workspace
  - resolve exact included files from baseline and round metadata
- `manifest.py`
  - build the package manifest payload
- `packager.py`
  - create the temporary `.tar.gz`
- `client.py`
  - perform HTTP upload and parse response payloads
- `workflow.py`
  - provide one shared high-level API used by both CLI entrypoints and optimize auto-upload

Add a new command module:

```text
src/helix/commands/upload_optimize.py
```

## CLI Wiring

Update:

- `src/helix/cli.py`
- `src/helix/commands/optimize.py`
- `src/helix/optimize/models.py`

Required behavior:

- add `upload-optimize`
- add `--no-upload` to `optimize`
- add `--no-upload` to `optimize-batch`
- plumb a boolean upload option into optimize runtime behavior

## Output Rules

### `upload-optimize`

Default output:

- print one short success summary on success
- print one short error summary on failure

Verbose output may additionally print:

- included file count
- excluded entry count
- tarball byte size
- upload destination
- parsed server response details

### Automatic Upload

For `optimize` and `optimize-batch`:

- no upload output by default
- on `--verbose`, print one short line for:
  - upload success
  - upload failure
  - skipped because `HELIX_OPTIMIZE_UPLOAD_URL` is unset

Automatic upload failures must not create extra local failure-record files in this design.

## Testing

Add tests for:

- CLI parser support for `upload-optimize`
- CLI parser support for `--no-upload` on `optimize` and `optimize-batch`
- `upload-optimize` fails when the upload URL is unset
- automatic upload skips silently when the URL is unset
- verbose mode reports skipped automatic upload when the URL is unset
- automatic upload triggers only after successful optimize completion
- automatic upload failure does not change optimize success exit codes
- whitelist inclusion for:
  - source and generated `.py` files
  - `opt-note.md`
  - `learned_lessons.md`
  - baseline state, operator, and perf artifact
  - round-local operator, notes, state, and perf artifact
  - `*.show-output.log`
- whitelist exclusion for:
  - `ir/`
  - `opt-verify/`
  - `.pt`
  - profiler output directories
  - `agent-sessions.jsonl`
- manifest generation
- tarball member layout
- request headers and content type
- response parsing and user-facing summaries

Run the standard repository verification commands documented in `README.md` after implementation.

## Documentation Updates

Update `README.md` to document:

- `upload-optimize`
- `--no-upload`
- `HELIX_OPTIMIZE_UPLOAD_URL`
- the high-level contract expected from the upload server under `services/helix-upload-server/`
