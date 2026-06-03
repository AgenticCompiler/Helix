# Incremental Show-Output Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make non-interactive `--show-output` logs durable during execution by writing rendered output chunks incrementally, stop storing full streamed output in `AgentResult.stdout`, and remove streamed-output duplication from optimize recovery and agent-invocation traces.

**Architecture:** Keep buffered and interactive execution unchanged. Add an incremental rendered-output sink to the streaming runner, wire `AgentRunner` and `OpenHandsRunner` to use it for `show_output=True`, add an explicit transient-retry flag on `AgentResult`, and update the few consumers that still assume streamed stdout is available after the run.

**Tech Stack:** Python 3, dataclasses, PTY/threaded process runners, existing output filters, `unittest`, `ruff`, `pyright`

---

## File Map

- Modify: `src/triton_agent/models.py`
  - Add an explicit result field for transient retry detection.
- Modify: `src/triton_agent/show_output_log.py`
  - Keep log-path helpers, replace marker-writing helpers with an incremental chunk writer.
- Modify: `src/triton_agent/process_runner.py`
  - Add streaming chunk sink support, optional streamed-stdout suppression, and rolling transient-failure detection for streaming runs.
- Modify: `src/triton_agent/backends/base.py`
  - Pass the show-output sink into streaming runs, stop post-run log writes, and switch retry logic to the explicit result flag for streamed runs.
- Modify: `src/triton_agent/backends/openhands.py`
  - Stream event text directly into the same log sink and stop returning aggregated stdout for `show_output=True`.
- Modify: `src/triton_agent/optimize/run_loop.py`
  - Replace stdout-derived stalled-recovery summaries with a fixed workspace-based continuation summary.
- Modify: `src/triton_agent/otel_trace.py`
  - Remove agent-invocation stdout/stderr digests and excerpts.
- Modify: `src/triton_agent/commands/report.py`
  - Stop depending on `result.stdout` for streamed failure messaging.
- Modify: `src/triton_agent/report/workspace.py`
  - Stop depending on `result.stdout` for streamed failure messaging.
- Modify: `src/triton_agent/log_check/log_check_launcher.py`
  - Stop depending on `result.stdout` for streamed failure messaging.
- Test: `tests/test_process_runner.py`
  - Streaming sink behavior, streamed-stdout suppression, transient retry detection.
- Test: `tests/test_backends_base.py`
  - Shared show-output log content and retry behavior.
- Test: `tests/test_openhands_runner.py`
  - Incremental log writes and empty streamed stdout.
- Test: `tests/test_supervisor.py`
  - Optimize stalled-recovery summary behavior.

### Task 1: Add Incremental Streaming Sink Support And Explicit Retry Metadata

**Files:**
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/show_output_log.py`
- Modify: `src/triton_agent/process_runner.py`
- Test: `tests/test_process_runner.py`

- [ ] **Step 1: Write the failing process-runner tests for incremental sink behavior**

```python
def test_streaming_can_write_rendered_chunks_to_incremental_sink(self) -> None:
    stdout = StringIO()
    sink = StringIO()
    process = _StreamingFakeProcess(wait_code=0)
    chunks = [b"line one\n", b"line two\n", b""]
    with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
        with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
            with patch(
                "triton_agent.process_runner.select.select",
                side_effect=[([11], [], []), ([11], [], []), ([11], [], [])],
            ):
                with patch("triton_agent.process_runner.os.read", side_effect=chunks):
                    with patch("triton_agent.process_runner.os.close"):
                        result = run_streaming_process(
                            ["codex", "exec"],
                            "/tmp",
                            stall_timeout_seconds=10,
                            stdout=stdout,
                            rendered_chunk_sink=sink.write,
                            collect_stdout=False,
                        )
    self.assertEqual(stdout.getvalue(), "line one\nline two\n")
    self.assertEqual(sink.getvalue(), "line one\nline two\n")
    self.assertEqual(result.stdout, "")
```

```python
def test_streaming_marks_retryable_failure_from_raw_chunks(self) -> None:
    process = _StreamingFakeProcess(wait_code=1)
    chunks = [b"ERROR: exceeded retry limit, ", b"last status: 429 Too Many Requests\n", b""]
    with patch("triton_agent.process_runner.pty.openpty", return_value=(11, 12)):
        with patch("triton_agent.process_runner.subprocess.Popen", return_value=process):
            with patch(
                "triton_agent.process_runner.select.select",
                side_effect=[([11], [], []), ([11], [], []), ([11], [], [])],
            ):
                with patch("triton_agent.process_runner.os.read", side_effect=chunks):
                    with patch("triton_agent.process_runner.os.close"):
                        result = run_streaming_process(
                            ["codex", "exec"],
                            "/tmp",
                            stall_timeout_seconds=10,
                            collect_stdout=False,
                        )
    self.assertTrue(result.retryable_failure)
```

- [ ] **Step 2: Run the targeted process-runner tests to verify they fail**

Run: `uv run python -m unittest tests.test_process_runner.StreamingProcessRunnerTests.test_streaming_can_write_rendered_chunks_to_incremental_sink tests.test_process_runner.StreamingProcessRunnerTests.test_streaming_marks_retryable_failure_from_raw_chunks -v`

Expected: FAIL because `run_streaming_process()` does not accept an incremental sink, still accumulates stdout, and does not expose a retryable-failure result field.

- [ ] **Step 3: Add the result field, log sink helper, and streaming-runner plumbing**

```python
# src/triton_agent/models.py
@dataclass
class AgentResult:
    return_code: int
    stdout: str
    stderr: str
    stalled: bool = False
    session_id: Optional[str] = None
    retryable_failure: bool = False
```

```python
# src/triton_agent/show_output_log.py
def write_show_output_chunk(stream: TextIO | None, text: str) -> None:
    if stream is None or not text:
        return
    stream.write(text)
    stream.flush()
```

```python
# src/triton_agent/process_runner.py
def run_streaming_process(
    command: list[str],
    workdir: str,
    stall_timeout_seconds: int,
    stdout: Optional[TextIO] = None,
    output_filter: Optional[OutputFilter] = None,
    session_id_extractor: Optional[Callable[[str], Optional[str]]] = None,
    interrupt_policy: Optional[InterruptPolicy] = None,
    extra_env: Optional[dict[str, str]] = None,
    *,
    rendered_chunk_sink: Callable[[str], None] | None = None,
    collect_stdout: bool = True,
) -> AgentResult:
    output_chunks: list[str] = []
    retryable_failure = False
    rolling_window = ""
    session_id: Optional[str] = None
    process = subprocess.Popen(
        _resolve_command(command),
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
        env=_merged_env(extra_env),
        start_new_session=interrupt_policy is not None,
    )
    if chunk:
        text = chunk.decode(errors="replace")
        rolling_window = (rolling_window + text.lower())[-4096:]
        retryable_failure = retryable_failure or _contains_transient_failure_text(rolling_window)
        filtered = output_filter.feed(text) if output_filter is not None else text
        if filtered:
            if collect_stdout:
                output_chunks.append(filtered)
            print(filtered, file=stdout or sys.stdout, end="")
            if rendered_chunk_sink is not None:
                rendered_chunk_sink(filtered)
        session_id = session_id or session_id_extractor(text)
    if output_filter is not None:
        trailing = output_filter.feed("", flush=True)
        if trailing:
            if collect_stdout:
                output_chunks.append(trailing)
            print(trailing, file=stdout or sys.stdout, end="")
            if rendered_chunk_sink is not None:
                rendered_chunk_sink(trailing)
    return AgentResult(
        return_code=process.wait(),
        stdout="".join(output_chunks) if collect_stdout else "",
        stderr="",
        stalled=False,
        session_id=session_id,
        retryable_failure=retryable_failure,
    )
```

- [ ] **Step 4: Run the focused process-runner test module**

Run: `uv run python -m unittest tests.test_process_runner -v`

Expected: `OK`

- [ ] **Step 5: Commit the streaming sink foundation**

```bash
git add src/triton_agent/models.py src/triton_agent/show_output_log.py src/triton_agent/process_runner.py tests/test_process_runner.py
git commit -m "feat: stream show-output logs incrementally"
```

### Task 2: Wire Base Backends And OpenHands To The Incremental Log Path

**Files:**
- Modify: `src/triton_agent/backends/base.py`
- Modify: `src/triton_agent/backends/openhands.py`
- Test: `tests/test_backends_base.py`
- Test: `tests/test_openhands_runner.py`

- [ ] **Step 1: Replace marker-based log assertions with stream-only expectations**

```python
def test_show_output_log_contains_streamed_text_without_wrapper_markers(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = _DummyRunner()
        request = AgentRequest(
            command_kind=CommandKind.GEN_TEST,
            input_path=workspace / "op.py",
            operator_path=workspace / "op.py",
            output_path=workspace / "test_op.py",
            test_mode=None,
            bench_mode=None,
            interact=False,
            verbose=False,
            show_output=True,
            force_overwrite=False,
            agent_name="dummy",
            skill_name="triton-npu-gen-test",
            prompt="Prompt body",
            workdir=workspace,
        )

        def _run_process(*args, **kwargs) -> AgentResult:
            sink = kwargs["rendered_chunk_sink"]
            sink("first streamed output\n")
            return AgentResult(return_code=0, stdout="", stderr="", session_id="session-1")

        with patch("triton_agent.backends.base.run_process", side_effect=_run_process):
            result = runner.run(request, stdout=StringIO())
        log_path = workspace / "triton-agent-logs" / "gen-test.show-output.log"
        content = log_path.read_text(encoding="utf-8")
        self.assertNotIn("triton-agent show-output start", content)
        self.assertNotIn("triton-agent show-output end", content)
        self.assertEqual(content, "first streamed output\n")
        self.assertEqual(result.stdout, "")
```

```python
def test_retry_uses_explicit_retryable_failure_for_show_output_runs(self) -> None:
    workspace = Path("/tmp")
    request = AgentRequest(
        command_kind=CommandKind.GEN_TEST,
        input_path=workspace / "op.py",
        operator_path=workspace / "op.py",
        output_path=workspace / "test_op.py",
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=False,
        show_output=True,
        force_overwrite=False,
        agent_name="dummy",
        skill_name="triton-npu-gen-test",
        prompt="Prompt body",
        workdir=workspace,
    )
    with (
        patch.dict(environ, {"TRITON_AGENT_CODE_AGENT_MAX_RETRIES": "1"}, clear=False),
        patch(
            "triton_agent.backends.base.run_process",
            side_effect=[
                AgentResult(return_code=1, stdout="", stderr="", retryable_failure=True),
                AgentResult(return_code=0, stdout="", stderr="", retryable_failure=False),
            ],
        ) as mocked_run,
        patch("time.sleep"),
    ):
        result = _DummyRunner().run(request)
    self.assertEqual(result.return_code, 0)
    self.assertEqual(mocked_run.call_count, 2)
```

```python
def test_show_output_writes_openhands_events_without_markers(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        runner = OpenHandsRunner()
        (workspace / ".openhands" / "skills").mkdir(parents=True)
        with patch(
            "triton_agent.backends.openhands._supports_openhands_runtime",
            return_value=True,
        ):
            with patch.dict(
                os.environ,
                {"LLM_API_KEY": "secret", "LLM_MODEL": "gpt-5.4-mini"},
                clear=True,
            ):
                with patch(
                    "triton_agent.backends.openhands._load_openhands_dependencies",
                    return_value=_fake_dependencies(),
                ):
                    result = runner.run(self._request(workspace, show_output=True), stdout=io.StringIO())
        log_path = workspace / "triton-agent-logs" / "gen-test.show-output.log"
    self.assertEqual(result.stdout, "")
    content = log_path.read_text(encoding="utf-8")
    self.assertNotIn("attempt=1", content)
    self.assertIn("assistant update", content)
    self.assertIn("assistant final", content)
```

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run: `uv run python -m unittest tests.test_backends_base.SharedRunnerBaseTests.test_show_output_log_contains_streamed_text_without_wrapper_markers tests.test_backends_base.SharedRunnerBaseTests.test_retry_uses_explicit_retryable_failure_for_show_output_runs tests.test_openhands_runner.OpenHandsRunnerTests.test_show_output_writes_event_text_to_workspace_log -v`

Expected: FAIL because base still writes marker text after the run, still retries by scanning stdout/stderr, and OpenHands still aggregates streamed stdout.

- [ ] **Step 3: Wire the shared runner and OpenHands to the incremental sink**

```python
# src/triton_agent/backends/base.py
with open_show_output_log(request) as log_stream:
    result = self._run_once(
        command,
        request,
        stdout=stdout,
        rendered_chunk_sink=(
            None if log_stream is None else lambda text: write_show_output_chunk(log_stream, text)
        ),
        collect_stdout=not request.show_output,
    )
```

```python
# src/triton_agent/backends/base.py
def _is_transient_agent_failure(result: AgentResult) -> bool:
    if result.stalled or result.return_code == 130:
        return False
    if result.retryable_failure:
        return True
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return any(pattern in combined for pattern in _TRANSIENT_AGENT_FAILURE_PATTERNS)
```

```python
# src/triton_agent/backends/openhands.py
with open_show_output_log(request) as log_stream:
    emitted_lines: list[str] = []

    def _capture_event(event: object) -> None:
        line = _event_to_text(event)
        if not line:
            return
        if request.show_output:
            rendered = line if line.endswith("\n") else f"{line}\n"
            print(rendered, file=stdout or sys.stdout, end="")
            write_show_output_chunk(log_stream, rendered)
        else:
            emitted_lines.append(line)
```

- [ ] **Step 4: Run the shared backend and OpenHands test modules**

Run: `uv run python -m unittest tests.test_backends_base tests.test_openhands_runner -v`

Expected: `OK`

- [ ] **Step 5: Commit the backend integration**

```bash
git add src/triton_agent/backends/base.py src/triton_agent/backends/openhands.py tests/test_backends_base.py tests/test_openhands_runner.py
git commit -m "feat: tee show-output streams directly to log files"
```

### Task 3: Remove Streamed-Stdout Dependencies From Optimize Recovery, Trace, And Failure Reporting

**Files:**
- Modify: `src/triton_agent/optimize/run_loop.py`
- Modify: `src/triton_agent/otel_trace.py`
- Modify: `src/triton_agent/commands/report.py`
- Modify: `src/triton_agent/report/workspace.py`
- Modify: `src/triton_agent/log_check/log_check_launcher.py`
- Test: `tests/test_supervisor.py`
- Test: `tests/test_report_command.py`
- Test: `tests/test_log_check_launcher.py`

- [ ] **Step 1: Write the failing recovery and failure-message tests**

```python
def test_stalled_retries_use_workspace_continuation_summary_not_stdout(self) -> None:
    request = AgentRequest(
        command_kind=CommandKind.OPTIMIZE,
        input_path=Path("/tmp/op.py"),
        operator_path=Path("/tmp/op.py"),
        output_path=Path("/tmp/opt_op.py"),
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=False,
        show_output=False,
        force_overwrite=False,
        agent_name="codex",
        skill_name="triton-npu-optimize",
        prompt="Optimize this operator",
        workdir=Path("/tmp"),
        min_rounds=1,
        round_mode="continuous",
    )
    runner = FakeRunner(
        [
            AgentResult(return_code=1, stdout="", stderr="", stalled=True),
            AgentResult(return_code=0, stdout="", stderr="", stalled=False),
        ]
    )
    result = OptimizeRunLoop(max_recovery_attempts=1).run(runner, request)
    self.assertEqual(result.return_code, 0)
    self.assertIn("previous invocation ended unexpectedly", runner.prompts[1].lower())
    self.assertNotIn("working...", runner.prompts[1])
```

```python
def test_handle_report_failure_with_show_output_uses_generic_log_hint(self) -> None:
    parser = build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp).resolve()
        args = parser.parse_args(["report", "-i", str(workspace), "--show-output"])
        stderr = io.StringIO()

        class FailingRunner:
            def run(self, request: AgentRequest) -> AgentResult:
                del request
                return AgentResult(return_code=1, stdout="", stderr="", stalled=False)

        with patch(
            "triton_agent.commands.report.resolve_staged_skills",
            side_effect=_dummy_resolve_staged_skills,
        ), patch(
            "triton_agent.commands.report.SkillLinkManager",
            return_value=_DummySkillLinkManager(),
        ), patch(
            "triton_agent.commands.report.create_runner",
            return_value=FailingRunner(),
        ), patch("sys.stderr", stderr):
            exit_code = handle_report(parser, args)

    self.assertEqual(exit_code, 1)
    self.assertIn("see show-output log", stderr.getvalue())
```

- [ ] **Step 2: Run the targeted recovery and reporting tests to verify they fail**

Run: `uv run python -m unittest tests.test_supervisor tests.test_report_command tests.test_log_check_launcher -v`

Expected: FAIL because stalled recovery still injects captured stdout into resume prompts and report/log-check failure messages still depend on `result.stdout`.

- [ ] **Step 3: Switch stalled recovery, trim trace payloads, and stop using streamed stdout for failure details**

```python
# src/triton_agent/optimize/run_loop.py
_STALL_RECOVERY_SUMMARY = (
    "The previous invocation ended unexpectedly before completion. "
    "Continue from the existing workspace state and recorded optimize artifacts."
)

while True:
    if resume_summary is None:
        result = runner.run(current_request)
    else:
        result = runner.resume(current_request, resume_summary)
    result, current_request = self._resume_until_round_requirement_satisfied(
        runner,
        current_request,
        result,
    )
    if result.succeeded:
        return result
    if current_request.interact:
        return result
    if not result.stalled or attempt >= self.max_recovery_attempts:
        return result
    attempt += 1
    resume_summary = _STALL_RECOVERY_SUMMARY
```

```python
# src/triton_agent/otel_trace.py
event: dict[str, Any] = {
    "schema_version": 1,
    "type": "agent_invocation",
    "phase": "end",
    "start_time": start_time,
    "end_time": end_time,
    "duration_ms": duration_ms,
    "status": status,
    "summary": summarize_agent_command(command, request.prompt),
    "return_code": return_code,
    "command_kind": request.command_kind.value,
    "agent": request.agent_name,
    "role": trace_role_from_request(request),
    "source": "runner",
    "confidence": "high",
}
```

```python
# src/triton_agent/report/workspace.py
if not result.succeeded:
    if request.show_output:
        return False, f"agent execution failed; see show-output log: {show_output_log_path(request)}"
    detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
    return False, detail[:120]
```

```python
# src/triton_agent/log_check/log_check_launcher.py
if not result.succeeded:
    if request.show_output:
        detail = f"agent execution failed; see show-output log: {show_output_log_path(request)}"
    else:
        detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
    print(f"[optimize-check] log check failed: {detail}", file=sys.stderr, flush=True)
    return result.return_code if result.return_code != 0 else 1
```

- [ ] **Step 4: Run the targeted regression set plus repository verification**

Run: `uv run python -m unittest tests.test_process_runner tests.test_backends_base tests.test_openhands_runner tests.test_supervisor tests.test_report_command tests.test_log_check_launcher -v`

Expected: `OK`

Run: `uv run --group dev ruff check`

Expected: `All checks passed!`

Run: `uv run pyright`

Expected: `0 errors, 0 warnings`

- [ ] **Step 5: Commit the contract cleanup**

```bash
git add src/triton_agent/optimize/run_loop.py src/triton_agent/otel_trace.py src/triton_agent/commands/report.py src/triton_agent/report/workspace.py src/triton_agent/log_check/log_check_launcher.py tests/test_supervisor.py tests/test_report_command.py tests/test_log_check_launcher.py
git commit -m "refactor: remove streamed stdout dependencies"
```
