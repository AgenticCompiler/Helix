from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any, TextIO

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult


class OpenHandsSetupError(RuntimeError):
    pass


@dataclass
class _OpenHandsDependencies:
    SecretStr: Any
    LLM: Any
    Tool: Any
    AgentContext: Any
    Agent: Any
    Conversation: Any
    LLMSummarizingCondenser: Any
    load_project_skills: Any
    NeverConfirm: Any
    TerminalTool: Any
    FileEditorTool: Any
    TaskTrackerTool: Any


class OpenHandsRunner(AgentRunner):
    def __init__(self, executable: str = "openhands", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        del request
        return [self.executable, "sdk"]

    def run(
        self,
        request: AgentRequest,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> AgentResult:
        if request.interact:
            return _error_result("OpenHands backend does not support --interact yet.")

        if not _supports_openhands_runtime():
            return _error_result("OpenHands backend requires Python 3.12 or newer.")

        api_key = os.environ.get("LLM_API_KEY", "").strip()
        if not api_key:
            return _error_result("OpenHands backend requires LLM_API_KEY to be set.")

        model = os.environ.get("LLM_MODEL", "").strip()
        if not model:
            return _error_result("OpenHands backend requires LLM_MODEL to be set.")

        if request.verbose:
            self._log_launch_command(self.build_command(request), stderr or sys.stderr)

        try:
            dependencies = _load_openhands_dependencies()
            conversation, emitted_lines = self._create_conversation(
                request,
                dependencies,
                model=model,
                api_key=api_key,
                stdout=stdout,
            )
            conversation.send_message(request.prompt)
            result = conversation.run()
        except OpenHandsSetupError as exc:
            return _error_result(str(exc))
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            return _error_result(f"OpenHands backend failed: {exc}")

        output = "\n".join(line for line in emitted_lines if line)
        if not output:
            final_line = _event_to_text(result)
            if final_line:
                output = final_line
        return AgentResult(return_code=0, stdout=output, stderr="")

    def _create_conversation(
        self,
        request: AgentRequest,
        dependencies: _OpenHandsDependencies,
        *,
        model: str,
        api_key: str,
        stdout: TextIO | None,
    ) -> tuple[Any, list[str]]:
        llm_kwargs: dict[str, object] = {
            "model": model,
            "api_key": dependencies.SecretStr(api_key),
        }
        base_url = os.environ.get("LLM_BASE_URL", "").strip()
        if base_url:
            llm_kwargs["base_url"] = _normalize_base_url(base_url)
        llm = dependencies.LLM(**llm_kwargs)

        skills_dir = request.workdir / ".openhands" / "skills"
        if not skills_dir.exists():
            raise OpenHandsSetupError(f"OpenHands skills path does not exist: {skills_dir}")

        project_skills = list(dependencies.load_project_skills(work_dir=str(request.workdir)))
        agent_context = dependencies.AgentContext(skills=project_skills)

        tools = [
            dependencies.Tool(name=dependencies.TerminalTool.name),
            dependencies.Tool(name=dependencies.FileEditorTool.name),
            dependencies.Tool(name=dependencies.TaskTrackerTool.name),
        ]
        agent = dependencies.Agent(
            llm=llm,
            tools=tools,
            agent_context=agent_context,
            condenser=dependencies.LLMSummarizingCondenser(llm=llm),
        )

        emitted_lines: list[str] = []

        def _capture_event(event: object) -> None:
            line = _event_to_text(event)
            if not line:
                return
            emitted_lines.append(line)
            if request.show_output:
                stream = stdout or sys.stdout
                print(line, file=stream)

        conversation = dependencies.Conversation(
            agent=agent,
            workspace=str(request.workdir),
            callbacks=[_capture_event],
        )
        conversation.set_confirmation_policy(dependencies.NeverConfirm())
        return conversation, emitted_lines


def _load_openhands_dependencies() -> _OpenHandsDependencies:
    try:
        sdk_module = import_module("openhands.sdk")
        condenser_module = import_module("openhands.sdk.context.condenser")
        skills_module = import_module("openhands.sdk.skills")
        security_module = import_module("openhands.sdk.security")
        terminal_module = import_module("openhands.tools.terminal")
        file_editor_module = import_module("openhands.tools.file_editor")
        task_tracker_module = import_module("openhands.tools.task_tracker")
        pydantic_module = import_module("pydantic")
    except ModuleNotFoundError as exc:
        raise OpenHandsSetupError(
            "OpenHands backend requires the openhands-sdk and openhands-tools packages to be installed."
        ) from exc

    return _OpenHandsDependencies(
        SecretStr=pydantic_module.SecretStr,
        LLM=sdk_module.LLM,
        Tool=sdk_module.Tool,
        AgentContext=sdk_module.AgentContext,
        Agent=sdk_module.Agent,
        Conversation=sdk_module.Conversation,
        LLMSummarizingCondenser=condenser_module.LLMSummarizingCondenser,
        load_project_skills=skills_module.load_project_skills,
        NeverConfirm=security_module.NeverConfirm,
        TerminalTool=terminal_module.TerminalTool,
        FileEditorTool=file_editor_module.FileEditorTool,
        TaskTrackerTool=task_tracker_module.TaskTrackerTool,
    )


def _event_to_text(event: object) -> str:
    if isinstance(event, str):
        return event.strip()
    for attribute in ("text", "content", "message", "value"):
        value = getattr(event, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    rendered = str(event).strip()
    return "" if rendered == object.__repr__(event) else rendered


def _error_result(message: str) -> AgentResult:
    return AgentResult(return_code=1, stdout="", stderr=message)


def _supports_openhands_runtime() -> bool:
    return sys.version_info >= (3, 12)


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized
