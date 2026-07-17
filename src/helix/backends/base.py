from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Optional, TextIO

from helix.backends.hook_common import HookStageOptions
from helix.models import AgentRequest, AgentResult
from helix.optimize.prompts import build_optimize_resume_prompt
from helix.trace.core import (
    TRACE_RUN_ID_ENV,
    append_trace_event,
    build_code_agent_event,
    trace_path_from_request,
    utc_timestamp,
)
from helix.backends.process_runner import InterruptPolicy, OutputFilter, run_process
from helix.terminal.logs import (
    open_show_output_log,
    write_show_output_chunk,
)
from helix.transient_failures import contains_transient_agent_failure_text
from helix.terminal.verbose import emit_command_block


_USER_PROMPT_PATH = "USER_PROMPT.md"


@contextmanager
def _windows_prompt_file(request: AgentRequest) -> Iterator[AgentRequest]:
    if sys.platform != "win32":
        yield request
        return

    prompt_path = request.workdir / _USER_PROMPT_PATH
    prompt_path.write_text(request.prompt, encoding="utf-8")
    try:
        launch_prompt = (
            "Read and follow the complete task instructions in "
            f"`{prompt_path.absolute()}` before doing any work."
        )
        yield request.with_prompt(launch_prompt)
    finally:
        if prompt_path.exists():
            prompt_path.unlink()


class AgentRunner(ABC):
    _OPTIMIZE_INTERRUPT_POLICY = InterruptPolicy()

    def __init__(self, executable: str, stall_timeout_seconds: int | None = None) -> None:
        self.executable = executable
        self.stall_timeout_seconds = (
            stall_timeout_seconds
            if stall_timeout_seconds is not None
            else _resolve_stall_timeout_seconds()
        )

    @abstractmethod
    def build_command(self, request: AgentRequest) -> list[str]:
        raise NotImplementedError

    def supports_mcp_servers(self) -> bool:
        return False

    @contextmanager
    def _prepare_run_context(
        self,
        request: AgentRequest,
        stderr: Optional[TextIO] = None,
    ) -> Iterator[None]:
        del request, stderr
        yield

    def run(
        self,
        request: AgentRequest,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        if request.mcp_servers and not self.supports_mcp_servers():
            return AgentResult(
                return_code=1,
                stdout="",
                stderr=f"{request.agent_name} backend does not support request-scoped MCP servers.",
            )
        with _windows_prompt_file(request) as launch_request:
            with self._prepare_run_context(launch_request, stderr=stderr):
                command = self.build_command(launch_request)
                if launch_request.verbose:
                    self._log_launch_command(command, stderr or sys.stderr)

                return self._run_with_retry(command, launch_request, stdout=stdout)

    def _extra_allowed_read_roots(self, request: AgentRequest) -> tuple[Path, ...]:
        if request.compiler_source_path is None:
            return ()
        return (request.compiler_source_path,)

    def interrupt_policy(self, request: AgentRequest) -> InterruptPolicy | None:
        if request.interact or request.command_kind != request.command_kind.OPTIMIZE:
            return None
        return self._OPTIMIZE_INTERRUPT_POLICY

    def resume(
        self,
        request: AgentRequest,
        summary: str,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        resumed_prompt = build_optimize_resume_prompt(
            summary,
            language=request.language,
            base_prompt=request.prompt,
            round_mode=request.round_mode,
            optimize_target=request.optimize_target,
            min_speedup=request.min_speedup,
            enable_subagent=request.enable_subagent,
        )
        return self.run(request.with_prompt(resumed_prompt), stdout=stdout, stderr=stderr)

    def session_id_extractor(self) -> Callable[[str], str | None]:
        return lambda _line: None

    def output_filter(self, request: AgentRequest) -> OutputFilter | None:
        del request
        return None

    def _log_launch_command(self, command: list[str], stream: TextIO) -> None:
        emit_command_block(stream, command)

    def _hook_options(self, request: AgentRequest) -> HookStageOptions:
        trace_path = trace_path_from_request(request)
        run_id = request.run_id or (request.extra_env or {}).get(TRACE_RUN_ID_ENV, "") if trace_path is not None else None
        return HookStageOptions(
            trace_enabled=request.log_tools and trace_path is not None,
            guard_enabled=request.enable_agent_hooks,
            trace_path=trace_path,
            run_id=run_id,
        )

    def _select_mode(self, request: AgentRequest) -> str:
        if request.interact:
            return "interactive"
        if request.stream_output:
            return "streaming"
        return "buffered"

    def _run_with_retry(
        self,
        command: list[str],
        request: AgentRequest,
        *,
        stdout: Optional[TextIO] = None,
    ) -> AgentResult:
        with open_show_output_log(request) as log_stream:
            rendered_chunk_sink: Callable[[str], None] | None = None
            if log_stream is not None:
                def _write_rendered_chunk(text: str) -> None:
                    write_show_output_chunk(log_stream, text)

                rendered_chunk_sink = _write_rendered_chunk

            collect_stdout = not request.stream_output
            result = self._run_once(
                command,
                request,
                stdout=stdout,
                rendered_chunk_sink=rendered_chunk_sink,
                collect_stdout=collect_stdout,
            )
            if request.interact:
                return result

            max_retries = _code_agent_max_retries()
            attempt = 0
            while (
                not request.disable_backend_retry
                and _is_transient_agent_failure(result)
                and attempt < max_retries
            ):
                attempt += 1
                time.sleep(_retry_delay_seconds(attempt))
                result = self._run_once(
                    command,
                    request,
                    stdout=stdout,
                    rendered_chunk_sink=rendered_chunk_sink,
                    collect_stdout=collect_stdout,
                )
            return result

    def _run_once(
        self,
        command: list[str],
        request: AgentRequest,
        *,
        stdout: Optional[TextIO] = None,
        rendered_chunk_sink: Callable[[str], None] | None = None,
        collect_stdout: bool = True,
    ) -> AgentResult:
        trace_path = trace_path_from_request(request)
        start_time = utc_timestamp()
        start_counter = time.perf_counter()
        result: AgentResult | None = None
        try:
            result = run_process(
                command,
                str(request.workdir),
                mode=self._select_mode(request),
                stall_timeout_seconds=self.stall_timeout_seconds,
                session_id_extractor=self.session_id_extractor(),
                stdout=stdout,
                output_filter=self.output_filter(request),
                interrupt_policy=self.interrupt_policy(request),
                extra_env=request.extra_env,
                rendered_chunk_sink=rendered_chunk_sink,
                collect_stdout=collect_stdout,
                progress_probe=request.progress_probe,
            )
            return result
        except BaseException as exc:
            end_time = utc_timestamp()
            duration_ms = int((time.perf_counter() - start_counter) * 1000)
            append_trace_event(
                trace_path,
                build_code_agent_event(
                    request=request,
                    command=command,
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    result=None,
                    exception=exc,
                ),
            )
            raise
        finally:
            if result is not None:
                end_time = utc_timestamp()
                duration_ms = int((time.perf_counter() - start_counter) * 1000)
                append_trace_event(
                    trace_path,
                    build_code_agent_event(
                        request=request,
                        command=command,
                        start_time=start_time,
                        end_time=end_time,
                        duration_ms=duration_ms,
                        result=result,
                    ),
                )

_CODE_AGENT_MAX_RETRIES_ENV = "HELIX_CODE_AGENT_MAX_RETRIES"
_DEFAULT_CODE_AGENT_MAX_RETRIES = 2
_STALL_TIMEOUT_SECONDS_ENV = "HELIX_STALL_TIMEOUT_SECONDS"
_DEFAULT_STALL_TIMEOUT_SECONDS = 900


def _is_transient_agent_failure(result: AgentResult) -> bool:
    if result.stalled or result.return_code == 130:
        return False
    if result.retryable_failure:
        return True
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return contains_transient_agent_failure_text(combined)


def _retry_delay_seconds(retry_number: int) -> float:
    return float(2 ** (retry_number - 1))


def _code_agent_max_retries() -> int:
    raw_value = os.environ.get(_CODE_AGENT_MAX_RETRIES_ENV)
    if raw_value is None:
        return _DEFAULT_CODE_AGENT_MAX_RETRIES
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_CODE_AGENT_MAX_RETRIES_ENV} must be a non-negative integer, got {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{_CODE_AGENT_MAX_RETRIES_ENV} must be a non-negative integer, got {raw_value!r}"
        )
    return value


def _resolve_stall_timeout_seconds() -> int:
    raw_value = os.environ.get(_STALL_TIMEOUT_SECONDS_ENV)
    if raw_value is None:
        return _DEFAULT_STALL_TIMEOUT_SECONDS
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_STALL_TIMEOUT_SECONDS_ENV} must be a non-negative integer, got {raw_value!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            f"{_STALL_TIMEOUT_SECONDS_ENV} must be a non-negative integer, got {raw_value!r}"
        )
    return value
