from __future__ import annotations

import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, TextIO

from triton_agent.agent_hooks import AgentHookManager
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.optimize.prompts import build_optimize_resume_prompt
from triton_agent.process_runner import InterruptPolicy, OutputFilter, run_process
from triton_agent.show_output_log import (
    open_show_output_log,
    write_show_output_attempt_result,
    write_show_output_attempt_start,
)
from triton_agent.verbose import emit_command_block, emit_verbose_lines


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

    def run(
        self,
        request: AgentRequest,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        command = self.build_command(request)
        if request.verbose:
            self._log_launch_command(command, stderr or sys.stderr)

        if not request.enable_agent_hooks:
            return self._run_with_retry(command, request, stdout=stdout)

        hook_manager = self._hook_manager()
        hook_state = hook_manager.prepare_hooks(request.agent_name, request.workdir)
        if request.verbose:
            emit_verbose_lines(stderr or sys.stderr, "hooks", hook_manager.describe_prepare(hook_state))
        try:
            return self._run_with_retry(command, request, stdout=stdout)
        finally:
            if request.verbose:
                emit_verbose_lines(stderr or sys.stderr, "hooks", hook_manager.describe_cleanup(hook_state))
            cleanup_warnings = hook_manager.cleanup(hook_state)
            if cleanup_warnings:
                emit_verbose_lines(stderr or sys.stderr, "hooks", cleanup_warnings)

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
            base_prompt=request.prompt,
            supervise=request.supervise,
        )
        return self.run(request.with_prompt(resumed_prompt), stdout=stdout, stderr=stderr)

    def session_id_extractor(self) -> Callable[[str], str | None]:
        return lambda _line: None

    def output_filter(self, request: AgentRequest) -> OutputFilter | None:
        del request
        return None

    def _log_launch_command(self, command: list[str], stream: TextIO) -> None:
        emit_command_block(stream, command)

    def _hook_manager(self) -> AgentHookManager:
        repo_root = Path(__file__).resolve().parents[3]
        return AgentHookManager(repo_root / "hooks")

    def _select_mode(self, request: AgentRequest) -> str:
        if request.interact:
            return "interactive"
        if request.show_output:
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
            attempt_number = 1
            write_show_output_attempt_start(log_stream, request=request, attempt_number=attempt_number)
            result = self._run_once(command, request, stdout=stdout)
            write_show_output_attempt_result(log_stream, result=result)
            if request.interact:
                return result

            max_retries = _code_agent_max_retries()
            attempt = 0
            while _is_transient_agent_failure(result) and attempt < max_retries:
                attempt += 1
                time.sleep(_retry_delay_seconds(attempt))
                attempt_number += 1
                write_show_output_attempt_start(log_stream, request=request, attempt_number=attempt_number)
                result = self._run_once(command, request, stdout=stdout)
                write_show_output_attempt_result(log_stream, result=result)
            return result

    def _run_once(
        self,
        command: list[str],
        request: AgentRequest,
        *,
        stdout: Optional[TextIO] = None,
    ) -> AgentResult:
        return run_process(
            command,
            str(request.workdir),
            mode=self._select_mode(request),
            stall_timeout_seconds=self.stall_timeout_seconds,
            session_id_extractor=self.session_id_extractor(),
            stdout=stdout,
            output_filter=self.output_filter(request),
            interrupt_policy=self.interrupt_policy(request),
            extra_env=request.extra_env,
        )


_TRANSIENT_AGENT_FAILURE_PATTERNS = (
    "429 too many requests",
    "exceeded retry limit",
    "rate limit",
)
_CODE_AGENT_MAX_RETRIES_ENV = "TRITON_AGENT_CODE_AGENT_MAX_RETRIES"
_DEFAULT_CODE_AGENT_MAX_RETRIES = 2
_STALL_TIMEOUT_SECONDS_ENV = "TRITON_AGENT_STALL_TIMEOUT_SECONDS"
_DEFAULT_STALL_TIMEOUT_SECONDS = 900


def _is_transient_agent_failure(result: AgentResult) -> bool:
    if result.stalled or result.return_code == 130:
        return False
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return any(pattern in combined for pattern in _TRANSIENT_AGENT_FAILURE_PATTERNS)


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
