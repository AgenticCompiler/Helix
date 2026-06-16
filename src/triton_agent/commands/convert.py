from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from triton_agent.commands.comparison import compare_result_files
from triton_agent.commands.input_resolution import resolve_single_operator_input
from triton_agent.convert.batch import resolve_batch_convert_operator_file, run_convert_batch
from triton_agent.convert.models import ConvertOptions
from triton_agent.convert.orchestration import build_convert_request, run_convert_request
from triton_agent.convert.outputs import prepare_convert_target
from triton_agent.execution import parse_test_metadata, run_local_test, run_remote_test
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.npu_affinity import resolve_batch_concurrency
from triton_agent.output import render_result
from triton_agent.paths import default_generated_output_path
from triton_agent.verbose import emit_verbose_lines

_MAX_CONVERT_AGENT_ATTEMPTS = 2


def _convert_test_label(test_mode: str) -> str:
    return "Standalone test file" if test_mode == "standalone" else "Differential test file"


@dataclass(frozen=True)
class _ConvertLoopResult:
    agent_result: AgentResult
    return_code: int
    validation_summary: str | None = None


@dataclass(frozen=True)
class _ConvertVerificationResult:
    return_code: int
    summary: str
    baseline_result: Path | None = None


def handle_convert(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    _validate_agent_options(parser, args)
    try:
        operator_path, workdir = resolve_single_operator_input(
            input_path,
            resolve_operator_file=resolve_batch_convert_operator_file,
        )
    except ValueError as exc:
        parser.error(str(exc))
    options = convert_options_from_args(args)
    request = build_convert_request(
        operator_path,
        operator_path,
        workdir,
        options,
    )
    output_path = request.output_path
    if output_path is None:
        parser.error("Internal error: convert request did not resolve an output path.")
    try:
        file_messages = prepare_convert_target(
            output_path,
            force_overwrite=options.force_overwrite,
        )
    except (FileExistsError, IsADirectoryError) as exc:
        parser.exit(2, f"{exc}\n")
    if options.verbose:
        emit_verbose_lines(sys.stderr, "files", file_messages)
    try:
        loop_result = _run_convert_with_verification_loop(request)
    except FileNotFoundError as exc:
        parser.error(
            f"Agent executable not found: {exc}. "
            f"Make sure the '{options.agent_name}' CLI is installed and available in PATH."
        )
    render_result(loop_result.agent_result, show_output=request.stream_output)
    if loop_result.validation_summary is not None:
        print(loop_result.validation_summary, file=sys.stderr)
    return loop_result.return_code


def handle_convert_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    try:
        max_concurrency = resolve_batch_concurrency(args.concurrency)
    except ValueError as exc:
        parser.error(str(exc))
    return run_convert_batch(
        root,
        convert_options_from_args(args),
        max_concurrency=max_concurrency,
    )


def _validate_agent_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if getattr(args, "agent", None) == "openhands" and bool(getattr(args, "interact", False)):
        parser.error("OpenHands backend does not support --interact yet.")


def convert_options_from_args(args: argparse.Namespace) -> ConvertOptions:
    return ConvertOptions(
        interact=bool(getattr(args, "interact", False)),
        verbose=bool(getattr(args, "verbose", False)),
        stream_output=bool(getattr(args, "stream_output", True)),
        force_overwrite=bool(getattr(args, "force_overwrite", False)),
        agent_name=args.agent,
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        prompt=getattr(args, "prompt", None),
        log_tools=bool(getattr(args, "log_tools", False)),
        enable_mcp=bool(getattr(args, "enable_mcp", False)),
    )


def _run_convert_with_verification_loop(request: AgentRequest) -> _ConvertLoopResult:
    base_prompt = request.prompt
    current_request = request
    cached_baseline_result: Path | None = None
    for attempt_index in range(_MAX_CONVERT_AGENT_ATTEMPTS):
        agent_result = run_convert_request(current_request)
        if agent_result.return_code != 0:
            return _ConvertLoopResult(agent_result=agent_result, return_code=agent_result.return_code)

        verification = _verify_converted_output(
            current_request,
            cached_baseline_result=cached_baseline_result,
        )
        if verification.baseline_result is not None:
            cached_baseline_result = verification.baseline_result
        if verification.return_code == 0:
            return _ConvertLoopResult(agent_result=agent_result, return_code=0)

        if attempt_index + 1 >= _MAX_CONVERT_AGENT_ATTEMPTS:
            return _ConvertLoopResult(
                agent_result=agent_result,
                return_code=verification.return_code,
                validation_summary=verification.summary,
            )

        current_request = current_request.with_prompt(
            _build_convert_repair_prompt(base_prompt, verification.summary)
        )

    raise AssertionError("convert verification loop exited without returning a result")


def _verify_converted_output(
    request: AgentRequest,
    *,
    cached_baseline_result: Path | None = None,
) -> _ConvertVerificationResult:
    converted_output = request.output_path
    if converted_output is None:
        return _ConvertVerificationResult(
            return_code=1,
            summary="Convert verification failed.\nMissing converted output path in request.",
        )
    if not converted_output.exists():
        return _ConvertVerificationResult(
            return_code=1,
            summary=(
                "Convert verification failed.\n"
                f"Converted operator does not exist: {converted_output}"
            ),
        )

    try:
        test_file = _resolve_convert_test_file(request)
    except (FileNotFoundError, ValueError) as exc:
        return _ConvertVerificationResult(
            return_code=1,
            summary=f"Convert verification failed.\n{exc}",
            baseline_result=cached_baseline_result,
        )

    test_mode = _resolve_convert_validation_mode(test_file, request.test_mode)
    if test_mode == "standalone":
        candidate_result, _candidate_archive, candidate_summary = _run_convert_validation_test(
            request,
            test_file,
            converted_output,
            role="converted output",
            test_mode="standalone",
        )
        if candidate_result.return_code != 0:
            return _ConvertVerificationResult(
                return_code=candidate_result.return_code,
                summary=candidate_summary,
                baseline_result=cached_baseline_result,
            )
        return _ConvertVerificationResult(
            return_code=0,
            summary=(
                "Convert verification passed.\n"
                f"Test file: {test_file}\n"
                f"Converted operator: {converted_output}"
            ),
            baseline_result=cached_baseline_result,
        )

    baseline_result = cached_baseline_result or _resolve_convert_baseline_result(request, test_file)
    if baseline_result is None:
        baseline_run_result, baseline_archive, baseline_summary = _run_convert_validation_test(
            request,
            test_file,
            request.input_path,
            role="original operator",
            test_mode="differential",
        )
        if baseline_run_result.return_code != 0:
            return _ConvertVerificationResult(
                return_code=baseline_run_result.return_code,
                summary=baseline_summary,
                baseline_result=None,
            )
        if baseline_archive is None:
            return _ConvertVerificationResult(
                return_code=1,
                summary=(
                    "Convert verification failed.\n"
                    f"{_convert_test_label(test_mode)}: {test_file}\n"
                    f"Original operator: {request.input_path}\n"
                    "Original operator differential run did not produce an archived result."
                ),
                baseline_result=None,
            )
        baseline_result = baseline_archive

    candidate_result, candidate_archive, candidate_summary = _run_convert_validation_test(
        request,
        test_file,
        converted_output,
        role="converted output",
        test_mode="differential",
    )
    if candidate_result.return_code != 0:
        return _ConvertVerificationResult(
            return_code=candidate_result.return_code,
            summary=candidate_summary,
            baseline_result=baseline_result,
        )
    if candidate_archive is None:
        return _ConvertVerificationResult(
            return_code=1,
            summary=(
                "Convert verification failed.\n"
                f"{_convert_test_label(test_mode)}: {test_file}\n"
                f"Converted operator: {converted_output}\n"
                "Converted-operator differential run did not produce an archived result."
            ),
            baseline_result=baseline_result,
        )

    compare_code = compare_result_files(baseline_result, candidate_archive)
    if compare_code != 0:
        return _ConvertVerificationResult(
            return_code=compare_code,
            summary=(
                "Convert verification failed.\n"
                f"{_convert_test_label(test_mode)}: {test_file}\n"
                f"Original operator: {request.input_path}\n"
                f"Converted operator: {converted_output}\n"
                f"Baseline result: {baseline_result}\n"
                f"Candidate result: {candidate_archive}\n"
                "Comparison result: compare-result failed."
            ),
            baseline_result=baseline_result,
        )

    return _ConvertVerificationResult(
        return_code=0,
        summary=(
            "Convert verification passed.\n"
            f"Test file: {test_file}\n"
            f"Converted operator: {converted_output}"
        ),
        baseline_result=baseline_result,
    )


def _resolve_convert_baseline_result(request: AgentRequest, test_file: Path) -> Path | None:
    expected_result = request.input_path.parent / f"{request.input_path.stem}_result.pt"
    if expected_result.exists():
        return expected_result.resolve()

    legacy_result = test_file.parent / "TEST_RESULT.pt"
    if legacy_result.exists():
        return legacy_result.resolve()

    return None


def _resolve_convert_test_file(request: AgentRequest) -> Path:
    preferred_mode = _normalize_convert_test_mode(request.test_mode)
    fallback_mode = "standalone" if preferred_mode == "differential" else "differential"
    preferred_default = default_generated_output_path(
        CommandKind.GEN_TEST,
        request.input_path,
        test_mode=preferred_mode,
    ).resolve()
    if preferred_default.exists():
        return preferred_default

    fallback_default = default_generated_output_path(
        CommandKind.GEN_TEST,
        request.input_path,
        test_mode=fallback_mode,
    ).resolve()
    if fallback_default.exists():
        return fallback_default

    preferred_candidates = _collect_convert_test_candidates(request.workdir, preferred_mode)
    if len(preferred_candidates) == 1:
        return preferred_candidates[0]
    if len(preferred_candidates) > 1:
        raise ValueError(
            _multiple_convert_test_candidates_message(
                request.workdir,
                preferred_mode,
                preferred_default,
            )
        )

    fallback_candidates = _collect_convert_test_candidates(request.workdir, fallback_mode)
    if len(fallback_candidates) == 1:
        return fallback_candidates[0]
    if len(fallback_candidates) > 1:
        raise ValueError(
            _multiple_convert_test_candidates_message(
                request.workdir,
                fallback_mode,
                fallback_default,
            )
        )

    raise FileNotFoundError(
        "convert verification could not find a reusable test file. "
        f"Expected {preferred_default}, {fallback_default}, exactly one differential_test_*.py file, "
        f"or exactly one test_*.py file in {request.workdir}."
    )


def _normalize_convert_test_mode(mode: str | None) -> str:
    return "standalone" if mode == "standalone" else "differential"


def _collect_convert_test_candidates(workdir: Path, test_mode: str) -> list[Path]:
    if test_mode == "standalone":
        return sorted(
            path.resolve()
            for path in workdir.iterdir()
            if path.is_file()
            and path.suffix == ".py"
            and path.name.startswith("test_")
            and not path.name.startswith("differential_test_")
        )
    return sorted(
        path.resolve()
        for path in workdir.iterdir()
        if path.is_file() and path.suffix == ".py" and path.name.startswith("differential_test_")
    )


def _multiple_convert_test_candidates_message(
    workdir: Path,
    test_mode: str,
    expected_default: Path,
) -> str:
    if test_mode == "standalone":
        return (
            "convert verification found multiple reusable standalone test files. "
            f"Expected {expected_default} or exactly one test_*.py file in {workdir}."
        )
    return (
        "convert verification found multiple reusable differential test files. "
        f"Expected {expected_default} or exactly one differential_test_*.py file in {workdir}."
    )


def _resolve_convert_validation_mode(test_file: Path, requested_mode: str | None = None) -> str:
    metadata = parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        return _normalize_convert_test_mode(requested_mode)
    return str(mode)


def _run_convert_validation_test(
    request: AgentRequest,
    test_file: Path,
    operator_file: Path,
    *,
    role: str,
    test_mode: str,
) -> tuple[AgentResult, Path | None, str]:
    try:
        if request.remote is not None:
            result, archived_result, _remote_workspace = run_remote_test(
                test_file,
                operator_file,
                test_mode,
                request.remote,
                request.remote_workdir,
                keep_remote_workdir=False,
                verbose=request.verbose,
                stderr=sys.stderr,
            )
        else:
            result, archived_result = run_local_test(
                test_file,
                operator_file,
                test_mode,
                verbose=request.verbose,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        summary = (
            "Convert verification failed.\n"
            f"{_convert_test_label(test_mode)}: {test_file}\n"
            f"{role.capitalize()}: {operator_file}\n"
            f"Validation runner error: {exc}"
        )
        return AgentResult(return_code=1, stdout="", stderr=str(exc)), None, summary

    if result.return_code != 0:
        summary = (
            "Convert verification failed.\n"
            f"{_convert_test_label(test_mode)}: {test_file}\n"
            f"{role.capitalize()}: {operator_file}\n"
            f"Validation run failed with return code {result.return_code}.\n"
            f"{_format_result_excerpt(result)}"
        )
        return result, archived_result, summary

    return result, archived_result, ""


def _format_result_excerpt(result: AgentResult) -> str:
    stdout = _truncate_text(result.stdout)
    stderr = _truncate_text(result.stderr)
    lines: list[str] = []
    if stdout:
        lines.extend(["Stdout:", stdout])
    if stderr:
        lines.extend(["Stderr:", stderr])
    if not lines:
        return "No stdout or stderr was captured."
    return "\n".join(lines)


def _truncate_text(text: str, *, limit: int = 1200) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rstrip() + "\n...[truncated]"


def _build_convert_repair_prompt(base_prompt: str, validation_summary: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "Follow-up convert verification failed.\n"
        "Use this follow-up failure context to repair the converted output or its reused/generated test, "
        "then rerun the same convert workflow expectations.\n\n"
        f"{validation_summary}"
    )
