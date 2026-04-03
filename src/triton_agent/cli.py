from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, TextIO, cast

from triton_agent.agent import AgentRunner
from triton_agent.claude_runner import ClaudeRunner
from triton_agent.codex_runner import CodexRunner
from triton_agent.models import AgentResult, COMMAND_TO_SKILL, AgentRequest, CommandKind
from triton_agent.optimize_guidance import OptimizeGuidanceManager
from triton_agent.opencode_runner import OpenCodeRunner
from triton_agent.paths import default_generated_output_path
from triton_agent.pi_runner import PiRunner
from triton_agent.prompts import build_prompt
from triton_agent.run_skill import load_run_skill_module
from triton_agent.skills import SkillLinkManager
from triton_agent.supervisor import OptimizeSupervisor
from triton_agent.verbose import emit_verbose, emit_verbose_lines


_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}
_RunSkillPayload = Mapping[str, object]


@dataclass(frozen=True)
class BatchOptimizeWorkspace:
    workspace: Path
    operator_file: Path


@dataclass(frozen=True)
class BatchOptimizeResult:
    workspace: Path
    succeeded: bool
    message: str


class _TestRunnerModule(Protocol):
    def run_local_test(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
    ) -> tuple[_RunSkillPayload, Path | None]: ...

    def run_remote_test(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        remote: str,
        remote_workdir: str | None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> tuple[_RunSkillPayload, Path | None, str]: ...

    def parse_test_metadata(self, test_file: Path) -> dict[str, str]: ...


class _CompareResultModule(Protocol):
    def compare_result_files(
        self, oracle_result: Path, new_result: Path, compare_level: str
    ) -> int: ...

    def compare_remote_result_files(
        self,
        oracle_result: Path,
        new_result: Path,
        compare_level: str,
        remote: str,
        remote_workdir: str | None,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> int: ...


class _BenchRunnerModule(Protocol):
    def run_local_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
    ) -> tuple[_RunSkillPayload, Path | None]: ...

    def run_remote_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        remote: str,
        remote_workdir: str | None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> tuple[_RunSkillPayload, Path | None, str]: ...

    def parse_bench_metadata(self, bench_file: Path) -> dict[str, str]: ...


class _ComparePerfModule(Protocol):
    def compare_perf_files(self, baseline_perf: Path, compare_perf: Path) -> int: ...


def _load_test_runner() -> _TestRunnerModule:
    return cast(_TestRunnerModule, load_run_skill_module("test_runner"))


def _load_compare_result() -> _CompareResultModule:
    return cast(_CompareResultModule, load_run_skill_module("compare_result"))


def _load_bench_runner() -> _BenchRunnerModule:
    return cast(_BenchRunnerModule, load_run_skill_module("bench_runner"))


def _load_compare_perf() -> _ComparePerfModule:
    return cast(_ComparePerfModule, load_run_skill_module("compare_perf"))


def run_local_test(test_file: Path, operator_file: Path, test_mode: str) -> tuple[AgentResult, Path | None]:
    result, archived = _load_test_runner().run_local_test(test_file, operator_file, test_mode)
    return _normalize_agent_result(result), archived


def run_remote_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[AgentResult, Path | None, str]:
    result, archived, remote_workspace = _load_test_runner().run_remote_test(
        test_file,
        operator_file,
        test_mode,
        remote,
        remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
    return _normalize_agent_result(result), archived, remote_workspace


def parse_test_metadata(test_file: Path) -> dict[str, str]:
    return _load_test_runner().parse_test_metadata(test_file)


def compare_result_files(oracle_result: Path, new_result: Path, compare_level: str) -> int:
    return _load_compare_result().compare_result_files(oracle_result, new_result, compare_level)


def compare_remote_result_files(
    oracle_result: Path,
    new_result: Path,
    compare_level: str,
    remote: str,
    remote_workdir: str | None,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    return _load_compare_result().compare_remote_result_files(
        oracle_result,
        new_result,
        compare_level,
        remote,
        remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def run_local_bench(
    bench_file: Path, operator_file: Path, bench_mode: str
) -> tuple[AgentResult, Path | None]:
    result, perf_path = _load_bench_runner().run_local_bench(bench_file, operator_file, bench_mode)
    return _normalize_agent_result(result), perf_path


def run_remote_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[AgentResult, Path | None, str]:
    result, perf_path, remote_workspace = _load_bench_runner().run_remote_bench(
        bench_file,
        operator_file,
        bench_mode,
        remote,
        remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )
    return _normalize_agent_result(result), perf_path, remote_workspace


def parse_bench_metadata(bench_file: Path) -> dict[str, str]:
    return _load_bench_runner().parse_bench_metadata(bench_file)


def compare_perf_files(baseline_perf: Path, compare_perf: Path) -> int:
    return _load_compare_perf().compare_perf_files(baseline_perf, compare_perf)


def _normalize_agent_result(result: AgentResult | _RunSkillPayload) -> AgentResult:
    if isinstance(result, AgentResult):
        return result
    payload = result
    required_keys = ("return_code", "stdout", "stderr")
    missing_keys = [key for key in required_keys if key not in payload]
    if missing_keys:
        raise ValueError(
            "Run skill result payload is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )
    session_id = payload.get("session_id")
    return AgentResult(
        return_code=int(str(payload["return_code"])),
        stdout=str(payload["stdout"]),
        stderr=str(payload["stderr"]),
        stalled=bool(payload.get("stalled", False)),
        session_id=None if session_id is None else str(session_id),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triton-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_kind in CommandKind:
        subparser = subparsers.add_parser(command_kind.value)
        subparser.set_defaults(command_kind=command_kind)
        if command_kind == CommandKind.RUN_TEST:
            subparser.add_argument("--test-file", required=True)
            subparser.add_argument("--operator-file", required=True)
        elif command_kind == CommandKind.RUN_BENCH:
            subparser.add_argument("--bench-file", required=True)
            subparser.add_argument("--operator-file", required=True)
        elif command_kind == CommandKind.COMPARE_RESULT:
            subparser.add_argument("--oracle-result", required=True)
            subparser.add_argument("--new-result", required=True)
            subparser.add_argument(
                "--compare-level",
                default="balanced",
                choices=["strict", "balanced", "relaxed"],
            )
        elif command_kind == CommandKind.COMPARE_PERF:
            subparser.add_argument("--baseline", required=True)
            subparser.add_argument("--compare", required=True)
        else:
            subparser.add_argument("-i", "--input", required=True)
        if command_kind in {
            CommandKind.GEN_TEST,
            CommandKind.GEN_BENCH,
            CommandKind.OPTIMIZE,
            CommandKind.OPTIMIZE_BATCH,
            CommandKind.RUN_TEST,
            CommandKind.RUN_BENCH,
            CommandKind.COMPARE_RESULT,
        }:
            subparser.add_argument("--remote")
            subparser.add_argument("--remote-workdir")
            if command_kind in {CommandKind.RUN_TEST, CommandKind.RUN_BENCH}:
                subparser.add_argument("--keep-remote-workdir", action="store_true")
        if command_kind not in {
            CommandKind.COMPARE_RESULT,
            CommandKind.COMPARE_PERF,
            CommandKind.OPTIMIZE_BATCH,
        }:
            subparser.add_argument("-o", "--output")
        if command_kind != CommandKind.COMPARE_PERF:
            subparser.add_argument("--verbose", action="store_true")
        if command_kind not in {CommandKind.COMPARE_RESULT, CommandKind.COMPARE_PERF}:
            if command_kind not in {
                CommandKind.RUN_TEST,
                CommandKind.RUN_BENCH,
                CommandKind.OPTIMIZE_BATCH,
            }:
                subparser.add_argument("--interact", action="store_true")
                subparser.add_argument("--show-output", action="store_true")
            if command_kind not in {CommandKind.RUN_TEST, CommandKind.RUN_BENCH}:
                subparser.add_argument(
                    "--agent", default="codex", choices=["codex", "opencode", "pi", "claude"]
                )
        if command_kind in {
            CommandKind.GEN_TEST,
            CommandKind.RUN_TEST,
            CommandKind.OPTIMIZE,
            CommandKind.OPTIMIZE_BATCH,
        }:
            subparser.add_argument(
                "--test-mode",
                default=(
                    "standalone"
                    if command_kind == CommandKind.GEN_TEST
                    else None
                ),
                choices=["standalone", "differential"],
            )
        if command_kind in {
            CommandKind.GEN_BENCH,
            CommandKind.RUN_BENCH,
            CommandKind.OPTIMIZE,
            CommandKind.OPTIMIZE_BATCH,
        }:
            subparser.add_argument(
                "--bench-mode",
                default=(
                    "standalone" if command_kind == CommandKind.GEN_BENCH else None
                ),
                choices=["standalone", "msprof"],
            )
        if command_kind in {CommandKind.OPTIMIZE, CommandKind.OPTIMIZE_BATCH}:
            subparser.add_argument("--min-rounds", type=int)
            subparser.add_argument("--continue", dest="continue_optimize", action="store_true")
            subparser.add_argument("--no-agent-session", action="store_true")
        if command_kind == CommandKind.OPTIMIZE_BATCH:
            subparser.add_argument("--max-concurrency", type=int, default=2)
        if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
            subparser.add_argument("--force-overwrite", action="store_true")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_command_aliases(argv))

    command_kind: CommandKind = args.command_kind
    if command_kind in {CommandKind.OPTIMIZE, CommandKind.OPTIMIZE_BATCH}:
        _validate_optimize_arguments(parser, args)
    if command_kind == CommandKind.OPTIMIZE_BATCH:
        return _run_optimize_batch(parser, args)
    continue_optimize = bool(getattr(args, "continue_optimize", False))

    if command_kind == CommandKind.COMPARE_RESULT:
        oracle_result = Path(args.oracle_result).expanduser().resolve()
        if not oracle_result.exists():
            parser.error(f"Oracle result path does not exist: {oracle_result}")
        new_result = Path(args.new_result).expanduser().resolve()
        if not new_result.exists():
            parser.error(f"New result path does not exist: {new_result}")
        if args.remote:
            try:
                return compare_remote_result_files(
                    oracle_result,
                    new_result,
                    args.compare_level,
                    args.remote,
                    args.remote_workdir,
                    verbose=args.verbose,
                    stderr=sys.stderr,
                )
            except (RuntimeError, ValueError) as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return compare_result_files(oracle_result, new_result, args.compare_level)
    if command_kind == CommandKind.COMPARE_PERF:
        baseline_perf = Path(args.baseline).expanduser().resolve()
        if not baseline_perf.exists():
            parser.error(f"Baseline perf path does not exist: {baseline_perf}")
        compare_perf = Path(args.compare).expanduser().resolve()
        if not compare_perf.exists():
            parser.error(f"Compare perf path does not exist: {compare_perf}")
        return compare_perf_files(baseline_perf, compare_perf)

    input_path, operator_path, workdir = _resolve_request_paths(parser, command_kind, args)
    if command_kind == CommandKind.RUN_TEST:
        resolved_test_mode = args.test_mode or _resolve_test_mode_from_metadata(input_path)
        remote_workspace: str | None = None
        try:
            if args.remote:
                result, archived_result, remote_workspace = run_remote_test(
                    input_path,
                    operator_path or input_path,
                    resolved_test_mode,
                    args.remote,
                    args.remote_workdir,
                    keep_remote_workdir=args.keep_remote_workdir,
                    verbose=args.verbose,
                    stderr=sys.stderr,
                )
            else:
                result, archived_result = run_local_test(
                    input_path,
                    operator_path or input_path,
                    resolved_test_mode,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        render_result(result, show_output=True)
        print(f"Return code: {result.return_code}")
        if archived_result is not None:
            print(f"Archived result: {archived_result}")
        if args.remote and args.keep_remote_workdir and remote_workspace is not None:
            print(f"Remote workspace: {remote_workspace}")
        return result.return_code
    if command_kind == CommandKind.RUN_BENCH:
        resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(input_path)
        remote_workspace: str | None = None
        try:
            if args.remote:
                result, perf_path, remote_workspace = run_remote_bench(
                    input_path,
                    operator_path or input_path,
                    resolved_bench_mode,
                    args.remote,
                    args.remote_workdir,
                    keep_remote_workdir=args.keep_remote_workdir,
                    verbose=args.verbose,
                    stderr=sys.stderr,
                )
            else:
                result, perf_path = run_local_bench(
                    input_path,
                    operator_path or input_path,
                    resolved_bench_mode,
                )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        render_result(result, show_output=True)
        print(f"Return code: {result.return_code}")
        if perf_path is not None:
            print(f"Perf file: {perf_path}")
        if args.remote and args.keep_remote_workdir and remote_workspace is not None:
            print(f"Remote workspace: {remote_workspace}")
        return result.return_code

    if command_kind == CommandKind.OPTIMIZE:
        try:
            request = _build_optimize_request(args, input_path, workdir)
        except ValueError as exc:
            parser.error(str(exc))
        result = _run_optimize_request(request)
        render_result(result, show_output=request.show_output)
        return result.return_code

    test_mode = getattr(args, "test_mode", None)
    bench_mode = getattr(args, "bench_mode", None)
    output_path = _resolve_output_path(command_kind, input_path, args.output, test_mode)
    force_overwrite = getattr(args, "force_overwrite", False)
    try:
        file_messages = prepare_generation_target(command_kind, output_path, force_overwrite)
    except (FileExistsError, IsADirectoryError) as exc:
        parser.exit(2, f"{exc}\n")
    if args.verbose:
        emit_verbose_lines(sys.stderr, "files", file_messages)
    prompt = build_prompt(
        command_kind,
        input_path,
        operator_path,
        output_path,
        test_mode,
        bench_mode,
        force_overwrite,
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
        getattr(args, "min_rounds", None),
        continue_optimize,
    )
    request = AgentRequest(
        command_kind=command_kind,
        input_path=input_path,
        operator_path=operator_path,
        output_path=output_path,
        test_mode=test_mode,
        bench_mode=bench_mode,
        interact=args.interact,
        verbose=args.verbose,
        show_output=args.show_output,
        force_overwrite=force_overwrite,
        agent_name=args.agent,
        skill_name=COMMAND_TO_SKILL[command_kind],
        prompt=prompt,
        workdir=workdir,
        min_rounds=getattr(args, "min_rounds", None),
        continue_optimize=continue_optimize,
        no_agent_session=bool(getattr(args, "no_agent_session", False)),
    )

    repo_root = Path(__file__).resolve().parents[2]
    manager = SkillLinkManager(repo_root / "skills")
    links = manager.prepare_skills(args.agent, workdir)
    if request.verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))
    try:
        runner = create_runner(args.agent)
        result = runner.run(request)
    finally:
        if request.verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(sys.stderr, "skills", warning)

    render_result(result, show_output=request.show_output)
    return result.return_code


def _validate_optimize_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.min_rounds is not None and args.min_rounds < 1:
        parser.error("--min-rounds must be at least 1")
    if getattr(args, "command_kind", None) == CommandKind.OPTIMIZE_BATCH:
        if args.max_concurrency < 1:
            parser.error("--max-concurrency must be at least 1")
    if getattr(args, "continue_optimize", False):
        if getattr(args, "test_mode", None) is not None:
            parser.error("--continue cannot be combined with --test-mode")
        if getattr(args, "bench_mode", None) is not None:
            parser.error("--continue cannot be combined with --bench-mode")


def _build_optimize_request(
    args: argparse.Namespace,
    input_path: Path,
    workdir: Path,
) -> AgentRequest:
    continue_optimize = bool(getattr(args, "continue_optimize", False))
    test_mode = getattr(args, "test_mode", None)
    bench_mode = getattr(args, "bench_mode", None)
    if continue_optimize:
        test_mode, bench_mode = _resolve_continue_optimize_modes(input_path, workdir)
    else:
        test_mode = test_mode or "differential"
        bench_mode = bench_mode or "standalone"
    output_path = _resolve_output_path(
        CommandKind.OPTIMIZE,
        input_path,
        getattr(args, "output", None),
        test_mode,
    )
    prompt = build_prompt(
        CommandKind.OPTIMIZE,
        input_path,
        input_path,
        output_path,
        test_mode,
        bench_mode,
        False,
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
        getattr(args, "min_rounds", None),
        continue_optimize,
    )
    return AgentRequest(
        command_kind=CommandKind.OPTIMIZE,
        input_path=input_path,
        operator_path=input_path,
        output_path=output_path,
        test_mode=test_mode,
        bench_mode=bench_mode,
        interact=bool(getattr(args, "interact", False)),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        force_overwrite=False,
        agent_name=args.agent,
        skill_name=COMMAND_TO_SKILL[CommandKind.OPTIMIZE],
        prompt=prompt,
        workdir=workdir,
        min_rounds=getattr(args, "min_rounds", None),
        continue_optimize=continue_optimize,
        no_agent_session=bool(getattr(args, "no_agent_session", False)),
    )


def _run_optimize_request(request: AgentRequest) -> AgentResult:
    repo_root = Path(__file__).resolve().parents[2]
    manager = SkillLinkManager(repo_root / "skills")
    links = manager.prepare_skills(request.agent_name, request.workdir)
    guidance_manager = OptimizeGuidanceManager()
    guidance_state = guidance_manager.prepare(
        request.workdir,
        request.input_path,
        test_mode=request.test_mode or "differential",
        bench_mode=request.bench_mode or "standalone",
    )
    if request.verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))
    if request.verbose:
        emit_verbose_lines(sys.stderr, "agents", guidance_manager.describe_prepare(guidance_state))
    try:
        runner = create_runner(request.agent_name)
        return OptimizeSupervisor().run(runner, request)
    finally:
        if request.verbose:
            emit_verbose_lines(sys.stderr, "agents", guidance_manager.describe_cleanup(guidance_state))
        warnings = guidance_manager.cleanup(guidance_state)
        for warning in warnings:
            emit_verbose(sys.stderr, "agents", warning)
        if request.verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(sys.stderr, "skills", warning)


def _run_optimize_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")

    workspace_candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not workspace_candidates:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    results: list[BatchOptimizeResult] = []
    runnable: list[BatchOptimizeWorkspace] = []
    for workspace in workspace_candidates:
        try:
            operator_file = _resolve_batch_optimize_operator_file(workspace)
        except ValueError as exc:
            results.append(BatchOptimizeResult(workspace=workspace, succeeded=False, message=str(exc)))
            continue
        runnable.append(BatchOptimizeWorkspace(workspace=workspace, operator_file=operator_file))

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures: dict[Future[AgentResult], BatchOptimizeWorkspace] = {}
        for item in runnable:
            try:
                request = _build_optimize_request(args, item.operator_file, item.workspace)
            except ValueError as exc:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=str(exc),
                    )
                )
                continue
            futures[executor.submit(_run_optimize_request, request)] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive boundary
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=f"unexpected optimize failure: {exc}",
                    )
                )
                continue
            if result.succeeded:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=True,
                        message=f"optimized {item.operator_file.name}",
                    )
                )
            else:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=_summarize_batch_optimize_failure(result),
                    )
                )

    return _render_batch_optimize_results(results)


def _resolve_batch_optimize_operator_file(workspace: Path) -> Path:
    candidates = [
        path
        for path in sorted(workspace.iterdir())
        if path.is_file() and _is_batch_optimize_operator_candidate(path)
    ]
    if not candidates:
        raise ValueError("found no candidate operator file after excluding generated artifacts")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"found multiple candidate operator files: {names}")
    return candidates[0]


def _is_batch_optimize_operator_candidate(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    if path.name in _BATCH_OPTIMIZE_EXCLUDED_NAMES:
        return False
    return not any(path.name.startswith(prefix) for prefix in _BATCH_OPTIMIZE_EXCLUDED_PREFIXES)


def _summarize_batch_optimize_failure(result: AgentResult) -> str:
    for output in (result.stderr, result.stdout):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"optimize exited with return code {result.return_code}"


def _render_batch_optimize_results(results: list[BatchOptimizeResult]) -> int:
    ordered_results = sorted(results, key=lambda item: item.workspace.name)
    succeeded = sum(1 for item in ordered_results if item.succeeded)
    failed = len(ordered_results) - succeeded
    for item in ordered_results:
        status = "OK" if item.succeeded else "FAIL"
        print(f"[{status}] {item.workspace.name}: {item.message}")
    print(f"Summary: {succeeded} succeeded, {failed} failed")
    return 0 if failed == 0 and ordered_results else 1


def _resolve_output_path(
    command_kind: CommandKind,
    input_path: Path,
    explicit_output: str | None,
    test_mode: str | None = None,
) -> Path | None:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH, CommandKind.OPTIMIZE}:
        return default_generated_output_path(command_kind, input_path, test_mode=test_mode)
    return None


def _resolve_request_paths(
    parser: argparse.ArgumentParser, command_kind: CommandKind, args: argparse.Namespace
) -> tuple[Path, Path | None, Path]:
    if command_kind == CommandKind.RUN_TEST:
        test_file = Path(args.test_file).expanduser().resolve()
        if not test_file.exists():
            parser.error(f"Test file path does not exist: {test_file}")
        operator_file = Path(args.operator_file).expanduser().resolve()
        if not operator_file.exists():
            parser.error(f"Operator file path does not exist: {operator_file}")
        return test_file, operator_file, test_file.parent

    if command_kind == CommandKind.RUN_BENCH:
        bench_file = Path(args.bench_file).expanduser().resolve()
        if not bench_file.exists():
            parser.error(f"Bench file path does not exist: {bench_file}")
        operator_file = Path(args.operator_file).expanduser().resolve()
        if not operator_file.exists():
            parser.error(f"Operator file path does not exist: {operator_file}")
        return bench_file, operator_file, bench_file.parent

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    return input_path, input_path, input_path.parent


def _normalize_command_aliases(argv: Optional[list[str]]) -> Optional[list[str]]:
    if argv is None or not argv:
        return argv
    aliases = {
        "gen_test": "gen-test",
        "run_test": "run-test",
        "gen_bench": "gen-bench",
        "run_bench": "run-bench",
        "compare_result": "compare-result",
        "compare_perf": "compare-perf",
        "optimize_batch": "optimize-batch",
    }
    normalized = list(argv)
    normalized[0] = aliases.get(normalized[0], normalized[0])
    return normalized


def _resolve_test_mode_from_metadata(test_file: Path) -> str:
    metadata = parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        raise ValueError(f"Test metadata is missing required 'test-mode' entry: {test_file}")
    return mode


def _resolve_bench_mode_from_metadata(bench_file: Path) -> str:
    metadata = parse_bench_metadata(bench_file)
    mode = metadata.get("bench-mode")
    if mode not in {"standalone", "msprof"}:
        raise ValueError(f"Benchmark metadata is missing required 'bench-mode' entry: {bench_file}")
    return mode


def _resolve_continue_optimize_modes(input_path: Path, workdir: Path) -> tuple[str, str]:
    opt_note_path = workdir / "opt-note.md"
    if not opt_note_path.exists():
        raise ValueError(f"Continue optimize requires existing opt-note.md: {opt_note_path}")
    if not any(path.is_dir() for path in workdir.glob("opt-round-*")):
        raise ValueError(
            f"Continue optimize requires at least one existing opt-round-* directory in {workdir}"
        )

    test_mode = _resolve_test_mode_from_metadata(_resolve_continue_test_harness(input_path))
    bench_mode = _resolve_bench_mode_from_metadata(_resolve_continue_bench_harness(input_path))
    return test_mode, bench_mode


def _resolve_continue_test_harness(input_path: Path) -> Path:
    candidates = [
        input_path.with_name(f"differential_test_{input_path.stem}.py"),
        input_path.with_name(f"test_{input_path.stem}.py"),
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise ValueError(
            f"Continue optimize requires an existing generated test harness for {input_path.name}"
        )
    if len(existing) > 1:
        raise ValueError(
            "Continue optimize found multiple test harnesses. Keep only the active optimize test harness."
        )
    return existing[0]


def _resolve_continue_bench_harness(input_path: Path) -> Path:
    harness = input_path.with_name(f"bench_{input_path.stem}.py")
    if not harness.exists():
        raise ValueError(
            f"Continue optimize requires an existing generated benchmark harness: {harness}"
        )
    return harness


def prepare_generation_target(
    command_kind: CommandKind, output_path: Path | None, force_overwrite: bool
) -> list[str]:
    if output_path is None:
        return []
    if command_kind not in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
        return []
    if not output_path.exists():
        return []
    if output_path.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory: {output_path}. Choose a file path instead."
        )
    if not force_overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force-overwrite to replace it."
        )
    # Remove the old artifact before launching the agent so generation starts from a
    # clean file instead of editing whatever happened to be there before.
    output_path.unlink()
    return [f"removed existing output file {output_path}"]


def render_result(
    result: AgentResult,
    show_output: bool,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> None:
    stdout_stream = stdout or sys.stdout
    stderr_stream = stderr or sys.stderr
    # `--show-output` already streamed stdout live, so printing it again here would
    # duplicate the transcript at the end of the run.
    if result.stdout and not show_output:
        print(result.stdout, file=stdout_stream, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=stderr_stream, end="" if result.stderr.endswith("\n") else "\n")


def create_runner(agent_name: str) -> AgentRunner:
    if agent_name == "codex":
        return CodexRunner()
    if agent_name == "opencode":
        return OpenCodeRunner()
    if agent_name == "pi":
        return PiRunner()
    if agent_name == "claude":
        return ClaudeRunner()
    raise ValueError(f"Unsupported agent backend: {agent_name}")


if __name__ == "__main__":
    raise SystemExit(main())
