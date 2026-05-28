from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from triton_agent.commands.convert import handle_convert, handle_convert_batch
from triton_agent.commands.comparison import handle_compare_perf, handle_compare_result
from triton_agent.commands.execution import handle_run_bench, handle_run_test
from triton_agent.commands.generation import handle_gen_bench, handle_gen_test
from triton_agent.commands.generation import handle_gen_eval
from triton_agent.commands.generation import handle_gen_eval_batch
from triton_agent.commands.status import handle_status
from triton_agent.commands.log_check import handle_log_check, handle_log_check_batch
from triton_agent.commands.verification import handle_verify, handle_verify_batch
from triton_agent.commands.optimize import (
    handle_optimize,
    handle_optimize_batch,
)
from triton_agent.commands.upload_optimize import handle_upload_optimize
from triton_agent.commands.report_batch import handle_report_batch
from triton_agent.commands.report import handle_report
from triton_agent.models import CommandKind


_Handler = Callable[[argparse.ArgumentParser, argparse.Namespace], int]
_AGENT_CHOICES = ("codex", "opencode", "pi", "claude", "openhands", "traecli")
_COMPARE_LEVEL_CHOICES = ("strict", "balanced", "relaxed")
_FORMAT_CHOICES = ("text", "markdown")
_TEST_MODE_CHOICES = ("standalone", "differential")
_BENCH_MODE_CHOICES = ("standalone", "msprof")
_RESUME_CHOICES = ("auto", "continue", "fresh")
_ROUND_MODE_CHOICES = ("continuous", "checked", "supervised")
_TARGET_CHIP_CHOICES = ("A3", "A5")
_OPTIMIZE_TARGET_CHOICES = ("kernel", "operator")
_OPTIMIZE_KNOWLEDGE_CHOICES = ("v1", "v2", "v3")
_VERIFY_PHASE_CHOICES = ("all", "test", "bench")
_TOP_LEVEL_DESCRIPTION = "Generate, run, verify, and optimize Triton NPU operator workflows."
_TOP_LEVEL_EXAMPLES = (
    "triton-agent gen-test -i kernel.py",
    "triton-agent convert -i kernel.py",
    "triton-agent convert-batch -i kernels",
    "triton-agent run-test --test-file test_kernel.py --operator-file kernel.py",
    "triton-agent compare-perf --baseline baseline.txt --compare candidate.txt",
    "triton-agent verify -i .",
    "triton-agent status -i .",
    "triton-agent log-check -i .",
    "triton-agent log-check-batch -i kernels",
    "triton-agent optimize -i kernel.py --agent codex",
    "triton-agent report-batch -i kernels",
)
_TOP_LEVEL_ENVIRONMENT_VARIABLE_GROUPS = (
    (
        "Batch and runtime",
        (
            (
                "TRITON_AGENT_BATCH_NPU_DEVICES",
                "Comma-separated Ascend NPU device pool for batch workspaces.",
            ),
            (
                "TRITON_AGENT_BATCH_WORKERS_PER_NPU",
                "Concurrent workers per NPU device in batch mode (positive int, default 1).",
            ),
            (
                "TRITON_AGENT_CODE_AGENT_MAX_RETRIES",
                "Retry limit for transient code-agent failures.",
            ),
            (
                "TRITON_AGENT_BENCH_OUTPUT_DIR",
                "Directory used to keep local benchmark profiler output.",
            ),
            (
                "TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES",
                "Enable ordinary optimize PT cleanup; default keeps PT files and does not affect --reset-optimize.",
            ),
            (
                "TRITON_AGENT_COMPILER_SOURCE_CACHE_DIR",
                "Overrides the base directory for cached compiler source checkouts. "
                "Defaults to ~/.triton-agent.",
            ),
            (
                "TRITON_AGENT_OPTIMIZE_UPLOAD_URL",
                "HTTP endpoint for the optimize upload server. "
                "Required for the upload-optimize subcommand and auto-upload after optimize.",
            ),
        ),
    ),
    (
        "Timeout control",
        (
            (
                "TRITON_AGENT_STALL_TIMEOUT_SECONDS",
                "Stall timeout in seconds for agent subprocesses (default: 900).",
            ),
            (
                "TRITON_AGENT_SSH_TIMEOUT_SECONDS",
                "Stall timeout in seconds for remote SSH operations (default: 120).",
            ),
            (
                "TRITON_AGENT_SCP_TIMEOUT_SECONDS",
                "Stall timeout in seconds for remote file transfers (default: 300).",
            ),
            (
                "TRITON_AGENT_EVAL_TIMEOUT_SECONDS",
                "Stall timeout in seconds for remote command execution (default: 900).",
            ),
            (
                "TRITON_AGENT_TEST_TIMEOUT_SECONDS",
                "Stall timeout in seconds for test execution (default: 900).",
            ),
            (
                "TRITON_AGENT_BENCH_TIMEOUT_SECONDS",
                "Stall timeout in seconds for benchmark execution (default: 900).",
            ),
            (
                "TRITON_AGENT_PROFILE_TIMEOUT_SECONDS",
                "Stall timeout in seconds for profile execution (default: 900).",
            ),
        ),
    ),
    (
        "OpenHands backend",
        (
            ("LLM_API_KEY", "Required API key for the OpenHands backend."),
            ("LLM_MODEL", "Required model name for the OpenHands backend."),
            ("LLM_BASE_URL", "Optional OpenHands API base URL."),
        ),
    ),
)


@dataclass(frozen=True)
class _CommandSpec:
    handler: _Handler
    help_group: str
    help_summary: str
    description: str
    input_mode: str = "input"
    has_output: bool = True
    has_verbose: bool = True
    has_remote: bool = False
    keep_remote_workdir: bool = False
    has_agent: bool = False
    has_interact: bool = False
    has_show_output: bool = False
    has_test_mode: bool = False
    test_mode_default: str | None = None
    test_mode_choices: tuple[str, ...] | None = None
    has_bench_mode: bool = False
    bench_mode_default: str | None = None
    has_npu_devices: bool = False
    has_optimize_options: bool = False
    has_prompt: bool = False
    max_concurrency_default: int | None = None
    report_workers_default: int | None = None
    has_force_overwrite: bool = False
    has_format: bool = False
    has_verify_phase: bool = False
    has_force_verify: bool = False
    has_log_tools: bool = False


_COMMAND_SPECS: dict[CommandKind, _CommandSpec] = {
    CommandKind.GEN_EVAL: _CommandSpec(
        handler=handle_gen_eval,
        help_group="Generation",
        help_summary="Generate test and benchmark harnesses for one operator.",
        description="Generate test and benchmark harnesses for one operator file.",
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="differential",
        has_bench_mode=True,
        bench_mode_default="standalone",
        has_prompt=True,
        has_force_overwrite=True,
        has_log_tools=True,
    ),
    CommandKind.GEN_EVAL_BATCH: _CommandSpec(
        handler=handle_gen_eval_batch,
        help_group="Generation",
        help_summary="Generate evaluation harnesses for multiple workspaces.",
        description="Generate evaluation harnesses for multiple operator workspaces.",
        has_output=False,
        has_remote=True,
        has_agent=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="differential",
        has_bench_mode=True,
        bench_mode_default="standalone",
        has_prompt=True,
        max_concurrency_default=2,
        has_log_tools=True,
    ),
    CommandKind.CONVERT: _CommandSpec(
        handler=handle_convert,
        help_group="Conversion",
        help_summary="Convert one PyTorch operator into a Triton NPU-backed PyTorch operator.",
        description="Convert one PyTorch operator file into a Triton NPU-backed PyTorch operator.",
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="differential",
        test_mode_choices=("differential",),
        has_prompt=True,
        has_force_overwrite=True,
        has_log_tools=True,
    ),
    CommandKind.CONVERT_BATCH: _CommandSpec(
        handler=handle_convert_batch,
        help_group="Conversion",
        help_summary="Convert multiple operator workspaces.",
        description="Convert multiple operator workspaces through the convert workflow.",
        has_output=False,
        has_remote=True,
        has_agent=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="differential",
        test_mode_choices=("differential",),
        has_prompt=True,
        max_concurrency_default=2,
        has_log_tools=True,
    ),
    CommandKind.GEN_TEST: _CommandSpec(
        handler=handle_gen_test,
        help_group="Generation",
        help_summary="Generate a test harness for one operator.",
        description="Generate a test harness for one operator file.",
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="standalone",
        has_prompt=True,
        has_force_overwrite=True,
        has_log_tools=True,
    ),
    CommandKind.RUN_TEST: _CommandSpec(
        handler=handle_run_test,
        help_group="Execution",
        help_summary="Run a generated test harness against an operator.",
        description="Run a generated test harness against one operator file.",
        input_mode="run-test",
        has_remote=True,
        keep_remote_workdir=True,
        has_test_mode=True,
    ),
    CommandKind.GEN_BENCH: _CommandSpec(
        handler=handle_gen_bench,
        help_group="Generation",
        help_summary="Generate a benchmark harness for one operator.",
        description="Generate a benchmark harness for one operator file.",
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_bench_mode=True,
        bench_mode_default="standalone",
        has_prompt=True,
        has_force_overwrite=True,
        has_log_tools=True,
    ),
    CommandKind.RUN_BENCH: _CommandSpec(
        handler=handle_run_bench,
        help_group="Execution",
        help_summary="Run a generated benchmark harness against an operator.",
        description="Run a generated benchmark harness against one operator file.",
        input_mode="run-bench",
        has_remote=True,
        keep_remote_workdir=True,
        has_bench_mode=True,
        has_npu_devices=True,
    ),
    CommandKind.COMPARE_RESULT: _CommandSpec(
        handler=handle_compare_result,
        help_group="Comparison",
        help_summary="Compare oracle and candidate result files.",
        description="Compare oracle and candidate result files for correctness.",
        input_mode="compare-result",
        has_output=False,
        has_remote=True,
    ),
    CommandKind.COMPARE_PERF: _CommandSpec(
        handler=handle_compare_perf,
        help_group="Comparison",
        help_summary="Compare baseline and candidate performance reports.",
        description="Compare baseline and candidate performance reports.",
        input_mode="compare-perf",
        has_output=False,
        has_verbose=False,
    ),
    CommandKind.STATUS: _CommandSpec(
        handler=handle_status,
        help_group="Status",
        help_summary="Show optimization status for one workspace.",
        description="Show optimization status for one workspace.",
        has_output=False,
        has_format=True,
    ),
    CommandKind.LOG_CHECK: _CommandSpec(
        handler=handle_log_check,
        help_group="Optimization",
        help_summary="Run Codex log strategy validation for one workspace.",
        description="Run Codex log validation and write structured JSON results.",
        has_output=False,
        has_agent=True,
        has_show_output=True,
        has_log_tools=True,
    ),
    CommandKind.LOG_CHECK_BATCH: _CommandSpec(
        handler=handle_log_check_batch,
        help_group="Optimization",
        help_summary="Run log strategy validation across multiple workspaces.",
        description="Run log strategy validation across multiple operator workspaces and write a root summary.",
        has_output=False,
        has_agent=True,
        has_show_output=True,
        max_concurrency_default=1,
        has_log_tools=True,
    ),
    CommandKind.VERIFY: _CommandSpec(
        handler=handle_verify,
        help_group="Verification",
        help_summary="Verify test and benchmark artifacts for one workspace.",
        description="Verify test and benchmark artifacts for one optimization workspace.",
        has_output=False,
        has_remote=True,
        keep_remote_workdir=True,
        has_test_mode=True,
        has_bench_mode=True,
        has_verify_phase=True,
    ),
    CommandKind.VERIFY_BATCH: _CommandSpec(
        handler=handle_verify_batch,
        help_group="Verification",
        help_summary="Verify artifacts for multiple optimization workspaces.",
        description="Verify artifacts for multiple optimization workspaces.",
        has_output=False,
        has_remote=True,
        keep_remote_workdir=True,
        has_force_verify=True,
    ),
    CommandKind.OPTIMIZE: _CommandSpec(
        handler=handle_optimize,
        help_group="Optimization",
        help_summary="Run the optimization workflow for one operator.",
        description="Run the optimization workflow for one operator file.",
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_test_mode=True,
        has_bench_mode=True,
        has_optimize_options=True,
        has_prompt=True,
        has_log_tools=True,
    ),
    CommandKind.OPTIMIZE_BATCH: _CommandSpec(
        handler=handle_optimize_batch,
        help_group="Optimization",
        help_summary="Run optimization across multiple workspaces.",
        description="Run optimization across multiple operator workspaces.",
        has_output=False,
        has_remote=True,
        has_agent=True,
        has_show_output=True,
        has_test_mode=True,
        has_bench_mode=True,
        has_optimize_options=True,
        has_prompt=True,
        max_concurrency_default=1,
        has_log_tools=True,
    ),
    CommandKind.UPLOAD_OPTIMIZE: _CommandSpec(
        handler=handle_upload_optimize,
        help_group="Optimization",
        help_summary="Upload one optimize workspace to an analysis server.",
        description="Upload one optimize workspace to an analysis server.",
        has_output=False,
        has_verbose=True,
    ),
    CommandKind.REPORT: _CommandSpec(
        handler=handle_report,
        help_group="Reporting",
        help_summary="Generate operator-level optimization report.",
        description="Launch an agent to read workspace artifacts and render report.md.",
        has_output=False,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_prompt=True,
    ),
    CommandKind.REPORT_BATCH: _CommandSpec(
        handler=handle_report_batch,
        help_group="Reporting",
        help_summary="Collect report-batch state and generate batch and per-workspace reports.",
        description="Scan a batch root, collect results into report-batch-state.json, render report-batch.md, "
                    "and generate per-workspace report.md via agent.",
        has_output=False,
        has_agent=True,
        has_show_output=True,
        report_workers_default=4,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triton-agent",
        usage="triton-agent [-h] COMMAND ...",
        description=_TOP_LEVEL_DESCRIPTION,
        epilog=_build_top_level_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    for command_kind in CommandKind:
        spec = _COMMAND_SPECS[command_kind]
        subparser = subparsers.add_parser(
            command_kind.value,
            help=spec.help_summary,
            description=spec.description,
        )
        subparser.set_defaults(command_kind=command_kind)
        _add_primary_arguments(subparser, spec)
        if spec.has_format:
            subparser.add_argument("--format", default="text", choices=_FORMAT_CHOICES)
        if spec.has_verify_phase:
            subparser.add_argument("--phase", default="all", choices=_VERIFY_PHASE_CHOICES)
        if spec.has_force_verify:
            subparser.add_argument("--force-verify", action="store_true")
        if spec.has_remote:
            subparser.add_argument("--remote")
            subparser.add_argument("--remote-workdir")
            if spec.keep_remote_workdir:
                subparser.add_argument("--keep-remote-workdir", action="store_true")
        if spec.has_output:
            subparser.add_argument("-o", "--output")
        if spec.has_verbose:
            subparser.add_argument("--verbose", action="store_true")
        if spec.has_interact:
            subparser.add_argument("--interact", action="store_true")
        if spec.has_show_output:
            subparser.add_argument("--show-output", action="store_true")
        if spec.has_log_tools:
            subparser.add_argument("--log-tools", action="store_true")
        if spec.has_agent:
            subparser.add_argument("--agent", default="codex", choices=_AGENT_CHOICES)
        if spec.has_test_mode:
            subparser.add_argument(
                "--test-mode",
                default=spec.test_mode_default,
                choices=spec.test_mode_choices or _TEST_MODE_CHOICES,
            )
        if spec.has_bench_mode:
            subparser.add_argument(
                "--bench-mode",
                default=spec.bench_mode_default,
                choices=_BENCH_MODE_CHOICES,
            )
        if spec.has_npu_devices:
            subparser.add_argument("--npu-devices")
        if spec.has_optimize_options:
            subparser.add_argument("--min-rounds", type=int, default=5)
            subparser.add_argument("--resume", default="auto", choices=_RESUME_CHOICES)
            subparser.add_argument("--reset-optimize", action="store_true")
            subparser.add_argument("--enable-compiler-source-analysis", action="store_true")
            subparser.add_argument("--enable-cann-ext-api", action="store_true")
            if command_kind == CommandKind.OPTIMIZE:
                subparser.add_argument("--enable-agent-hooks", action="store_true")
            subparser.add_argument("--target-chip", default="A5", choices=_TARGET_CHIP_CHOICES)
            subparser.add_argument(
                "--optimize-target",
                default="kernel",
                choices=_OPTIMIZE_TARGET_CHOICES,
            )
            subparser.add_argument(
                "--optimize-knowledge",
                default="v1",
                choices=_OPTIMIZE_KNOWLEDGE_CHOICES,
            )
            subparser.add_argument("--no-agent-session", action="store_true")
            subparser.add_argument(
                "--round-mode",
                default="continuous",
                choices=_ROUND_MODE_CHOICES,
            )
            subparser.add_argument("--no-upload", action="store_true")
            subparser.add_argument("--report", action="store_true", default=True)
        if spec.has_prompt:
            subparser.add_argument("--prompt")
        if command_kind in {CommandKind.LOG_CHECK, CommandKind.LOG_CHECK_BATCH}:
            subparser.add_argument(
                "--check-result-file",
                default="log_check_result.json",
                help="Workspace-relative log check result JSON file name.",
            )
        if command_kind == CommandKind.LOG_CHECK_BATCH:
            subparser.add_argument(
                "--summary-file",
                default="log_check_summary.md",
                help="Root-relative batch log check summary file name.",
            )
        if spec.max_concurrency_default is not None:
            subparser.add_argument("--max-concurrency", type=int, default=spec.max_concurrency_default)
        if spec.report_workers_default is not None:
            subparser.add_argument("--report-workers", type=int, default=spec.report_workers_default)
        if spec.has_force_overwrite:
            subparser.add_argument("--force-overwrite", action="store_true")

    return parser


def _build_top_level_epilog() -> str:
    lines = ["Command groups:"]
    group_names = (
        "Generation",
        "Conversion",
        "Execution",
        "Comparison",
        "Status",
        "Verification",
        "Optimization",
        "Reporting",
    )
    for group_name in group_names:
        lines.append(f"{group_name}:")
        for command_kind in CommandKind:
            spec = _COMMAND_SPECS[command_kind]
            if spec.help_group == group_name:
                lines.append(f"  {command_kind.value:<22} {spec.help_summary}")
    lines.append("")
    lines.append("Examples:")
    for example in _TOP_LEVEL_EXAMPLES:
        lines.append(f"  {example}")
    lines.append("")
    lines.extend(_build_environment_variables_section())
    return "\n".join(lines)


def _build_environment_variables_section() -> list[str]:
    lines = ["Environment variables:"]
    for group_name, entries in _TOP_LEVEL_ENVIRONMENT_VARIABLE_GROUPS:
        lines.append(f"{group_name}:")
        for name, description in entries:
            lines.append(f"  {name:<36} {description}")
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_command_aliases(argv))
    command_kind = args.command_kind
    return _COMMAND_SPECS[command_kind].handler(parser, args)


def _add_primary_arguments(subparser: argparse.ArgumentParser, spec: _CommandSpec) -> None:
    if spec.input_mode == "run-test":
        subparser.add_argument("--test-file", required=True)
        subparser.add_argument("--operator-file", required=True)
        subparser.add_argument("--oracle-result")
        subparser.add_argument("--compare-level", choices=_COMPARE_LEVEL_CHOICES)
        return
    if spec.input_mode == "run-bench":
        subparser.add_argument("--bench-file", required=True)
        subparser.add_argument("--operator-file", required=True)
        return
    if spec.input_mode == "compare-result":
        subparser.add_argument("--oracle-result", required=True)
        subparser.add_argument("--new-result", required=True)
        subparser.add_argument("--compare-level", default="balanced", choices=_COMPARE_LEVEL_CHOICES)
        return
    if spec.input_mode == "compare-perf":
        subparser.add_argument("--baseline", required=True)
        subparser.add_argument("--compare", required=True)
        subparser.add_argument("--skip-latency-errors", action="store_true")
        subparser.add_argument(
            "--metric-source",
            default="auto",
            choices=("auto", "kernel", "total-op", "all"),
        )
        return
    subparser.add_argument("-i", "--input", required=True)


def _normalize_command_aliases(argv: Optional[list[str]]) -> Optional[list[str]]:
    if argv is None or not argv:
        return argv
    aliases = {
        "gen_eval": "gen-eval",
        "gen_eval_batch": "gen-eval-batch",
        "convert_batch": "convert-batch",
        "gen_test": "gen-test",
        "run_test": "run-test",
        "gen_bench": "gen-bench",
        "run_bench": "run-bench",
        "compare_result": "compare-result",
        "compare_perf": "compare-perf",
        "verify_batch": "verify-batch",
        "optimize_batch": "optimize-batch",
        "log_check": "log-check",
        "log_check_batch": "log-check-batch",
        "report_batch": "report-batch",
        "report": "report",
    }
    normalized = list(argv)
    normalized[0] = aliases.get(normalized[0], normalized[0])
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
