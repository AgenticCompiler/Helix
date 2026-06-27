from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from triton_agent.models import AgentRequest


def show_output_log_path(request: AgentRequest) -> Path:
    run_id = request.run_id
    label = request.show_output_label
    if run_id:
        base = request.workdir / "triton-agent-logs" / run_id
        if label:
            return base / f"show-output-{label}.log"
        return base / "show-output.log"
    return request.workdir / "triton-agent-logs" / f"{request.command_kind.value}.show-output.log"


@contextmanager
def open_show_output_log(request: AgentRequest) -> Iterator[TextIO | None]:
    if request.interact or not request.stream_output:
        yield None
        return
    path = show_output_log_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        yield stream


def write_show_output_chunk(stream: TextIO | None, text: str) -> None:
    if stream is None or not text:
        return
    stream.write(text)
    stream.flush()
