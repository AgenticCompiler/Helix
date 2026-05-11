from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.resources import application_root, skills_root


_PATTERNS_SUBDIR: tuple[str, ...] = ("triton-npu-optimize-knowledge", "references", "patterns")


def build_log_check_prompt(*, target_path: Path, output_file: str = "log_check_result.md") -> str:
    normalized_target = target_path.resolve()
    output_path = normalized_target / output_file
    patterns_dir = skills_root().joinpath(*_PATTERNS_SUBDIR)
    patterns_relative = Path(os.path.relpath(patterns_dir, start=application_root())).as_posix()
    return f"""\
Please examine the following directory:

  {normalized_target.as_posix()}

This directory contains optimization records where opt-round-i holds the i-th round.
Each round directory includes:
  - round-state.json  : optimization round state
  - summary.md        : round summary
  - attempts/         : attempted strategies for the round
  - *.py              : optimized kernel code
  - perf.txt          : performance results after optimization

Also examine the patterns reference directory:

  {patterns_relative}

This directory contains the full set of optimization strategies provided to the
agent. The pattern index file lists all available strategies.

Perform the following checks and write the complete analysis results to the operator
root directory file {output_file}.

check-1: Each optimization round uses a distinct strategy
Analyze whether each optimization round attempts a different strategy. Compare the
approaches used across all opt-round-i directories.
result: PASS
detail: If passing, list each round's strategy. If failing, mark which rounds reused
the same strategy.

check-2: Strategy novelty beyond provided patterns
Analyze whether the optimization log contains strategies beyond those listed in the
patterns reference directory.
If new strategies are found:
  result: PASS
  detail: Which rounds used novel strategies
If no new strategies are found:
  result: FAIL
  detail: Only the following patterns from the reference were used

check-3: Parameter tuning should use autotune instead of many manual rounds
Check whether the optimization process spent many rounds only manually tuning
parameters (block size, num_warps, num_stages, tile size, etc.) without using or
attempting to use autotune. If there are consecutive or numerous rounds that only
adjust these parameters manually, mark as FAIL.
result: PASS
detail: If passing, note which rounds used autotune or did not have excessive manual
parameter tuning. If failing, list the rounds that repeatedly performed manual
parameter tuning and suggest using autotune instead.

check-4: Optimized code must not duplicate or regress to previous rounds
Verify that the optimized code in each round does not duplicate or regress to code
from earlier rounds. Compare the *.py files across opt-round-i directories and confirm
that each round builds on the latest optimization, not copying the previous round,
regressing to an earlier round, or making only meaningless formatting changes.
result: PASS
detail: If passing, describe the key change in each round relative to the previous
one. If failing, mark which rounds duplicated, regressed, or had no substantive change.

check-6: Triton invocation must remain in use
Verify that each round still calls the Triton kernel path and has not replaced it with
pure PyTorch implementation. The optimized code must not remove, bypass, or weaken the
Triton invocation to pass tests or boost performance by avoiding Triton.
result: PASS
detail: If passing, describe the Triton call path preserved in each round. If failing,
mark which rounds removed, bypassed, or weakened the Triton invocation.

check-7: Baseline operator correctness and benchmark are valid
The baseline directory under the operator root records the original operator version.
Verify based on baseline state, original operator snapshot, test results, benchmark
output, and perf records.
Check that the baseline operator passes all tests and that benchmark results are
correct and reasonable.
result: PASS
detail: If passing, describe the test status, benchmark status, and performance result
source in the baseline. If failing, note missing or failed baseline files, test
failures, benchmark failures, or unreasonable performance results.

check-8: Best optimized version is valid and verified
The opt-note file records the optimization process, including what the agent considers
the best version. First locate the best version from opt-note, round-state, summary, or
related logs, then check its optimized code, test results, benchmark results, and perf
records in the corresponding operator directory.
Verify that the best version identified by the agent is obtainable, tests pass, and
benchmark results are correct and reasonable.
result: PASS
detail: If passing, state which round is the best version, evidence source, test
status, benchmark status, and performance results. If failing, note inability to
confirm the best version, test failures, benchmark failures, or unreasonable
performance results.

check-9: Round logs and evidence files are complete
Verify that each optimization round has saved all necessary log and evidence files.
Each round should ideally contain: optimization plan or attempt records, optimization
summary, optimized code, performance results, msprof output summary (if msprof was
run), and records of compilation fixes and runtime error handling. Judge based on
attempts/, summary.md, round-state.json, perf.txt, and profile/msprof-related files.
result: PASS
detail: If passing, describe per-round what logs and evidence were saved. If failing,
mark which rounds are missing optimization plans, optimized code, msprof summaries,
compilation fix records, or runtime error records.

Output format requirements:

The file must begin with a check overview in the following format:

summary:
overall: PASS or FAIL
failed_checks: none (if overall PASS) or list of check numbers and titles with result FAIL
overview_detail: A brief paragraph summarizing the overall conclusion and main risks

Overall result rule: overall is PASS only when ALL check sections have result PASS.
If any check is FAIL, overall must be FAIL.

Must include all eight sections: check-1, check-2, check-3, check-4, check-6, check-7,
check-8, and check-9.
Each section must contain the check title, result, and detail.
result must be exactly PASS or FAIL — no other casing or values are allowed.
Organize detail by section; avoid single-sentence conclusions.

Write the final result directly to: {output_path.as_posix()}"""


def build_log_check_request(
    *,
    target_path: Path,
    workdir: Path,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    output_file: str = "log_check_result.md",
) -> AgentRequest:
    resolved_workdir = workdir.resolve()
    resolved_target = target_path.resolve()
    return AgentRequest(
        command_kind=CommandKind.LOG_CHECK,
        input_path=resolved_target,
        operator_path=None,
        output_path=None,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=verbose,
        show_output=show_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="triton-npu-optimize-check",
        prompt=build_log_check_prompt(target_path=resolved_target, output_file=output_file),
        workdir=resolved_workdir,
        no_agent_session=True,
    )


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog_name or Path(__file__).name,
        description="Launch Codex log validation and write log_check_result.md.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Operator workspace root path containing baseline and opt-round-* directories.",
    )
    parser.add_argument(
        "--output-file",
        default="log_check_result.md",
        help="Output filename written in the target workspace (default: log_check_result.md).",
    )
    return parser


def run_log_check(
    *,
    target_path: Path,
    output_file: str = "log_check_result.md",
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
) -> int:
    normalized_target = target_path.expanduser().resolve()
    repo_root = application_root()
    request = build_log_check_request(
        target_path=normalized_target,
        workdir=repo_root,
        agent_name=agent_name,
        verbose=verbose,
        show_output=show_output,
        output_file=output_file,
    )
    try:
        runner = create_runner(agent_name)
    except ValueError as exc:
        print(f"[optimize-check] invalid agent: {exc}", file=sys.stderr, flush=True)
        return 2

    print(
        "[optimize-check] start log check: "
        + (
            f"path={normalized_target.as_posix()}, workdir={repo_root.as_posix()}, "
            f"output={output_file}, agent={agent_name}"
        ),
        file=sys.stderr,
        flush=True,
    )
    try:
        result = runner.run(request)
    except FileNotFoundError as exc:
        print(
            f"[optimize-check] agent executable not found: {exc}. "
            f"Make sure the '{agent_name}' CLI is installed and available in PATH.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if not result.succeeded:
        detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
        print(f"[optimize-check] log check failed: {detail}", file=sys.stderr, flush=True)
        return result.return_code if result.return_code != 0 else 1

    output_path = normalized_target / output_file
    if not output_path.is_file():
        print(
            "[optimize-check] log check completed but output file was not created: "
            + output_path.as_posix(),
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        "[optimize-check] log check completed: " + output_path.as_posix(),
        file=sys.stderr,
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    parser = build_parser(prog_name=prog_name)
    args = parser.parse_args(argv)
    return run_log_check(
        target_path=Path(args.path),
        output_file=str(args.output_file),
        verbose=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
