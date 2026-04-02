from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, TextIO

from triton_agent.codex_runner import CodexRunner
from triton_agent.models import AgentResult, COMMAND_TO_SKILL, AgentRequest, CommandKind
from triton_agent.optimize_guidance import OptimizeGuidanceManager
from triton_agent.opencode_runner import OpenCodeRunner
from triton_agent.paths import default_generated_output_path
from triton_agent.prompts import build_prompt
from triton_agent.run_skill import load_run_skill_module
from triton_agent.skills import SkillLinkManager
from triton_agent.supervisor import OptimizeSupervisor
from triton_agent.verbose import emit_verbose, emit_verbose_lines


def run_local_test(*args, **kwargs):
    result, archived = load_run_skill_module("test_runner").run_local_test(*args, **kwargs)
    return _normalize_agent_result(result), archived


def run_remote_test(*args, **kwargs):
    result, archived, remote_workspace = load_run_skill_module("test_runner").run_remote_test(
        *args, **kwargs
    )
    return _normalize_agent_result(result), archived, remote_workspace


def parse_test_metadata(*args, **kwargs):
    return load_run_skill_module("test_runner").parse_test_metadata(*args, **kwargs)


def compare_result_files(*args, **kwargs):
    return load_run_skill_module("compare_result").compare_result_files(*args, **kwargs)


def compare_remote_result_files(*args, **kwargs):
    return load_run_skill_module("compare_result").compare_remote_result_files(*args, **kwargs)


def run_local_bench(*args, **kwargs):
    result, perf_path = load_run_skill_module("bench_runner").run_local_bench(*args, **kwargs)
    return _normalize_agent_result(result), perf_path


def run_remote_bench(*args, **kwargs):
    result, perf_path, remote_workspace = load_run_skill_module("bench_runner").run_remote_bench(
        *args, **kwargs
    )
    return _normalize_agent_result(result), perf_path, remote_workspace


def parse_bench_metadata(*args, **kwargs):
    return load_run_skill_module("bench_runner").parse_bench_metadata(*args, **kwargs)


def compare_perf_files(*args, **kwargs):
    return load_run_skill_module("compare_perf").compare_perf_files(*args, **kwargs)


def _normalize_agent_result(result) -> AgentResult:
    if isinstance(result, AgentResult):
        return result
    return AgentResult(
        return_code=int(result["return_code"]),
        stdout=str(result["stdout"]),
        stderr=str(result["stderr"]),
        stalled=bool(result.get("stalled", False)),
        session_id=result.get("session_id"),
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
            CommandKind.RUN_TEST,
            CommandKind.RUN_BENCH,
            CommandKind.COMPARE_RESULT,
        }:
            subparser.add_argument("--remote")
            subparser.add_argument("--remote-workdir")
            if command_kind in {CommandKind.RUN_TEST, CommandKind.RUN_BENCH}:
                subparser.add_argument("--keep-remote-workdir", action="store_true")
        if command_kind not in {CommandKind.COMPARE_RESULT, CommandKind.COMPARE_PERF}:
            subparser.add_argument("-o", "--output")
        if command_kind != CommandKind.COMPARE_PERF:
            subparser.add_argument("--verbose", action="store_true")
        if command_kind not in {CommandKind.COMPARE_RESULT, CommandKind.COMPARE_PERF}:
            if command_kind not in {CommandKind.RUN_TEST, CommandKind.RUN_BENCH}:
                subparser.add_argument("--interact", action="store_true")
                subparser.add_argument("--show-output", action="store_true")
            if command_kind not in {CommandKind.RUN_TEST, CommandKind.RUN_BENCH}:
                subparser.add_argument(
                    "--agent", default="codex", choices=["codex", "opencode"]
                )
        if command_kind in {CommandKind.GEN_TEST, CommandKind.RUN_TEST, CommandKind.OPTIMIZE}:
            subparser.add_argument(
                "--test-mode",
                default=(
                    "differential"
                    if command_kind == CommandKind.OPTIMIZE
                    else "standalone" if command_kind == CommandKind.GEN_TEST else None
                ),
                choices=["standalone", "differential"],
            )
        if command_kind in {CommandKind.GEN_BENCH, CommandKind.RUN_BENCH, CommandKind.OPTIMIZE}:
            subparser.add_argument(
                "--bench-mode",
                default="standalone" if command_kind != CommandKind.RUN_BENCH else None,
                choices=["standalone", "msprof"],
            )
        if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
            subparser.add_argument("--force-overwrite", action="store_true")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_command_aliases(argv))

    command_kind: CommandKind = args.command_kind
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
        if args.remote and args.keep_remote_workdir:
            print(f"Remote workspace: {remote_workspace}")
        return result.return_code
    if command_kind == CommandKind.RUN_BENCH:
        resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(input_path)
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
        if args.remote and args.keep_remote_workdir:
            print(f"Remote workspace: {remote_workspace}")
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
    )

    repo_root = Path(__file__).resolve().parents[2]
    manager = SkillLinkManager(repo_root / "skills")
    links = manager.prepare_skills(args.agent, workdir)
    guidance_manager = OptimizeGuidanceManager()
    guidance_state = None
    if request.verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))
    if command_kind == CommandKind.OPTIMIZE:
        guidance_state = guidance_manager.prepare(
            workdir,
            input_path,
            test_mode=test_mode or "differential",
            bench_mode=bench_mode or "standalone",
        )
        if request.verbose:
            emit_verbose_lines(sys.stderr, "agents", guidance_manager.describe_prepare(guidance_state))
    try:
        runner = create_runner(args.agent)
        if command_kind == CommandKind.OPTIMIZE:
            result = OptimizeSupervisor().run(runner, request)
        else:
            result = runner.run(request)
    finally:
        if guidance_state is not None:
            if request.verbose:
                emit_verbose_lines(
                    sys.stderr, "agents", guidance_manager.describe_cleanup(guidance_state)
                )
            warnings = guidance_manager.cleanup(guidance_state)
            for warning in warnings:
                emit_verbose(sys.stderr, "agents", warning)
        if request.verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(sys.stderr, "skills", warning)

    render_result(result, show_output=request.show_output)
    return result.return_code


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
    result, show_output: bool, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None
) -> None:
    stdout_stream = stdout or sys.stdout
    stderr_stream = stderr or sys.stderr
    # `--show-output` already streamed stdout live, so printing it again here would
    # duplicate the transcript at the end of the run.
    if result.stdout and not show_output:
        print(result.stdout, file=stdout_stream, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=stderr_stream, end="" if result.stderr.endswith("\n") else "\n")


def create_runner(agent_name: str):
    if agent_name == "codex":
        return CodexRunner()
    if agent_name == "opencode":
        return OpenCodeRunner()
    raise ValueError(f"Unsupported agent backend: {agent_name}")


if __name__ == "__main__":
    raise SystemExit(main())
