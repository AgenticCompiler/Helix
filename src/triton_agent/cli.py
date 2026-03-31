from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, TextIO

from triton_agent.codex_runner import CodexRunner
from triton_agent.models import COMMAND_TO_SKILL, AgentRequest, CommandKind
from triton_agent.optimize_guidance import OptimizeGuidanceManager
from triton_agent.opencode_runner import OpenCodeRunner
from triton_agent.paths import default_generated_output_path
from triton_agent.prompts import build_prompt
from triton_agent.skills import SkillLinkManager
from triton_agent.supervisor import OptimizeSupervisor
from triton_agent.verbose import emit_verbose, emit_verbose_lines


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
        else:
            subparser.add_argument("-i", "--input", required=True)
        subparser.add_argument("-o", "--output")
        subparser.add_argument("--interact", action="store_true")
        subparser.add_argument("--verbose", action="store_true")
        subparser.add_argument("--show-output", action="store_true")
        subparser.add_argument("--agent", default="codex", choices=["codex", "opencode"])
        if command_kind in {CommandKind.GEN_TEST, CommandKind.RUN_TEST, CommandKind.OPTIMIZE}:
            subparser.add_argument(
                "--test-mode",
                default="differential" if command_kind == CommandKind.OPTIMIZE else "standalone",
                choices=["standalone", "differential"],
            )
        if command_kind in {CommandKind.GEN_BENCH, CommandKind.RUN_BENCH, CommandKind.OPTIMIZE}:
            subparser.add_argument(
                "--bench-mode", default="standalone", choices=["standalone", "msprof"]
            )
        if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
            subparser.add_argument("--force-overwrite", action="store_true")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_command_aliases(argv))

    command_kind: CommandKind = args.command_kind
    input_path, operator_path, workdir = _resolve_request_paths(parser, command_kind, args)
    output_path = _resolve_output_path(command_kind, input_path, args.output)
    force_overwrite = getattr(args, "force_overwrite", False)
    test_mode = getattr(args, "test_mode", None)
    bench_mode = getattr(args, "bench_mode", None)
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
    command_kind: CommandKind, input_path: Path, explicit_output: str | None
) -> Path | None:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH, CommandKind.OPTIMIZE}:
        return default_generated_output_path(command_kind, input_path)
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
    }
    normalized = list(argv)
    normalized[0] = aliases.get(normalized[0], normalized[0])
    return normalized


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
