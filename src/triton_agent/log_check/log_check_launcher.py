from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.otel_trace import build_tool_trace_env, new_trace_run_id, trace_path_from_request, write_tool_trace_summary
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.show_output_log import show_output_log_path
from triton_agent.skills import SkillLinkManager, staged_skill_dir
from triton_agent.verbose import emit_verbose_lines

from .check_json import (
    repair_json,
    validate_log_check_json,
    validate_pattern_analysis_json,
)
from .render_markdown import (
    render_log_check_markdown,
    render_pattern_analysis_markdown,
)

_LOG_CHECK_JSON_FILENAME = "log_check_result.json"
_PATTERN_ANALYSIS_JSON_FILENAME = "pattern_analysis.json"

_LOG_CHECK_JSON_SCHEMA_EXAMPLE = r"""{
  "schema_version": 1,
  "overall": "PASS",
  "failed_checks": "none",
  "overview_detail": "A brief paragraph summarizing the overall conclusion and main risks.",
  "checks": [
    {
      "id": "check-1",
      "name": "distinct strategies per round",
      "result": "pass",
      "detail": "Each round used a different strategy: round-1 applied tiling, ..."
    },
    {
      "id": "check-2",
      "name": "strategy novelty beyond patterns",
      "result": "fail",
      "detail": "Only used patterns already in the reference directory..."
    }
  ]
}"""

_PATTERN_ANALYSIS_JSON_SCHEMA_EXAMPLE = r"""{
  "schema_version": 1,
  "rounds": [
    {
      "round": "round-1",
      "patterns": [
        {
          "name": "tiling",
          "evidence": "explicit",
          "source": "round-1/attempts.md: 'applied tiling pattern to improve memory access'"
        }
      ]
    }
  ],
  "summary": {
    "given": [
      { "name": "tiling", "rounds": [1, 2], "evidence": "explicit" }
    ],
    "new": [
      { "name": "host-side shape dispatch", "rounds": [3] }
    ],
    "extended": [
      { "name": "tile budget tuning", "rounds": [5], "from": "tiling" }
    ]
  }
}"""


def build_log_check_prompt(
    *,
    target_path: Path,
    log_check_json_file: str = _LOG_CHECK_JSON_FILENAME,
    pattern_analysis_json_file: str = _PATTERN_ANALYSIS_JSON_FILENAME,
    agent_name: str = "codex",
    language: Literal["triton", "tilelang"] = "triton",
) -> str:
    normalized_target = target_path.resolve()
    backend_skills = staged_skill_dir(agent_name)
    patterns_path = backend_skills / f"{language}-npu-optimize-knowledge" / "references" / "patterns"
    return f"""\
Please examine your current working directory (this is the operator workspace):

  {normalized_target.as_posix()}

This directory contains optimization records where opt-round-i holds the i-th round.
Each round directory includes:
  - round-state.json  : optimization round state
  - summary.md        : round summary
  - attempts/         : attempted strategies for the round
  - *.py              : optimized kernel code
  - perf.txt          : performance results after optimization

Also examine the patterns reference directory staged under your workspace:

  {patterns_path.as_posix()}

This directory contains the full set of optimization strategies provided to the
agent. The pattern index file lists all available strategies.

Perform the following checks and write the complete analysis results as structured
JSON files.

--- Check definitions ---

check-1: Each optimization round uses a distinct strategy
Analyze whether each optimization round attempts a different strategy. Compare the
approaches used across all opt-round-i directories.

check-2: Strategy novelty beyond provided patterns
Analyze whether the optimization log contains strategies beyond those listed in the
patterns reference directory.
If new strategies are found: result pass. If no new strategies are found: result fail.

check-3: Parameter tuning should use autotune instead of many manual rounds
Check whether the optimization process spent many rounds only manually tuning
parameters (block size, num_warps, num_stages, tile size, etc.) without using or
attempting to use autotune. If there are consecutive or numerous rounds that only
adjust these parameters manually, mark as fail.

check-4: Optimized code must not duplicate or regress to previous rounds
Verify that the optimized code in each round does not duplicate or regress to code
from earlier rounds. Compare the *.py files across opt-round-i directories and confirm
that each round builds on the latest optimization, not copying the previous round,
regressing to an earlier round, or making only meaningless formatting changes.

check-6: Triton invocation must remain in use
Verify that each round still calls the Triton kernel path and has not replaced it with
pure PyTorch implementation. The optimized code must not remove, bypass, or weaken the
Triton invocation to pass tests or boost performance by avoiding Triton.

check-7: Baseline operator correctness and benchmark are valid
The baseline directory under the operator root records the original operator version.
Verify based on baseline state, original operator snapshot, test results, benchmark
output, and perf records.
Check that the baseline operator passes all tests and that benchmark results are
correct and reasonable.

check-8: Best optimized version is valid and verified
The opt-note file records the optimization process, including what the agent considers
the best version. First locate the best version from opt-note, round-state, summary, or
related logs, then check its optimized code, test results, benchmark results, and perf
records in the corresponding operator directory.
Verify that the best version identified by the agent is obtainable, tests pass, and
benchmark results are correct and reasonable.

check-9: Round logs and evidence files are complete
Verify that each optimization round has saved all necessary log and evidence files.
Each round should ideally contain: optimization plan or attempt records, optimization
summary, optimized code, performance results, msprof output summary (if msprof was
run), and records of compilation fixes and runtime error handling. Judge based on
attempts/, summary.md, round-state.json, perf.txt, and profile/msprof-related files.

check-10: Optimization pattern usage analysis

Analyze which optimization patterns from the staged pattern reference were applied in
each optimization round. This check is informational — it reports findings without a
PASS/FAIL result. Write the analysis to a separate file: {pattern_analysis_json_file}

For each round, use two-tier evidence in priority order:

Tier 1 — Explicit (read artifacts first):
  - Search opt-round-N/attempts.md for pattern names, candidate pattern discussions,
    and pattern triage records.
  - Search opt-round-N/summary.md for named pattern direction records.
  - Also check opt-note.md at the operator root for overall pattern mentions.

Tier 2 — Inferred (only when Tier 1 finds no pattern for a round):
  - Compare the operator .py file between this round and its immediate predecessor
    (previous round or baseline/).
  - Read each pattern's ## Signals section from the staged pattern references under
    {patterns_path.as_posix()}
  - Match the nature of the code diff against pattern signals.

For each detected pattern, label its evidence level:
  - "explicit": pattern name directly stated in attempts.md, summary.md, or opt-note.md.
    Cite the source file and round.
  - "inferred": determined from code diff analysis. Cite the key diff changes and
    which pattern signals matched.

--- Output format ---

Write TWO JSON files (NOT markdown files):

1. {log_check_json_file} — check results for check-1 through check-9.
   Schema:
{_LOG_CHECK_JSON_SCHEMA_EXAMPLE}

   Rules:
   - "schema_version" must be 1.
   - "overall" is "PASS" ONLY when ALL eight check sections have result "pass".
     If any check has result "fail", overall must be "FAIL".
   - "failed_checks": when overall is "PASS", set to "none".
     When overall is "FAIL", list the check ids and titles that failed, e.g.
     "check-2: strategy novelty beyond patterns, check-4: no code duplication or regression".
   - "overview_detail": a brief paragraph summarizing the overall conclusion and main risks.
   - checks array: exactly 8 entries (check-1, check-2, check-3, check-4, check-6, check-7, check-8, check-9).
     DO NOT include check-5 or check-10 in this file.
   - checks[].id: the check identifier string, e.g. "check-1".
   - checks[].name: the check title, e.g. "distinct strategies per round".
   - checks[].result: "pass" or "fail" (lowercase).
   - checks[].detail: string describing the evidence. Use null when there is nothing to say.

2. {pattern_analysis_json_file} — pattern usage analysis (check-10 only).
   Schema:
{_PATTERN_ANALYSIS_JSON_SCHEMA_EXAMPLE}

   Rules:
   - "schema_version" must be 1.
   - "rounds": per-round pattern detection (one entry per opt-round-N directory).
   - "rounds[].patterns[].evidence": "explicit" or "inferred".
   - "rounds[].patterns[].source": citation string for the evidence.
   - "summary.given": all patterns matched against staged references, grouped by name
      with the rounds they appeared in and their evidence level.
   - "summary.new": any strategies that do not match ANY staged pattern reference.
   - "summary.extended": strategies that build on a given pattern but add a new
      variation. The "from" field names the base pattern.

CRITICAL — JSON formatting requirements:
- Write valid JSON. Escape double quotes inside strings as \\\", escape newlines as \\n,
  escape backslashes as \\\\.
- Do NOT use trailing commas.
- Do NOT wrap the JSON in markdown code fences (```json ... ```).
- Write directly to the file using a write_file tool, not as displayed output.
- Each file must be a single JSON object at the top level.
- Ensure all strings are properly closed. Multi-paragraph detail strings are fine
  as long as newlines are escaped.

Write the check results directly to: {log_check_json_file}
Write the pattern analysis directly to: {pattern_analysis_json_file}"""


def build_log_check_request(
    *,
    target_path: Path,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    output_json: str = _LOG_CHECK_JSON_FILENAME,
    pattern_analysis_json: str = _PATTERN_ANALYSIS_JSON_FILENAME,
    staged_skill_names: tuple[str, ...] | None = None,
    staged_skill_sources: dict[str, str] | None = None,
    log_tools: bool = False,
    language: Literal["triton", "tilelang"] = "triton",
) -> AgentRequest:
    resolved_target = target_path.resolve()
    extra_env = None
    run_id = new_trace_run_id(prefix="log-check")
    if log_tools:
        extra_env, _trace_path, _ = build_tool_trace_env(None, workdir=resolved_target, run_id=run_id)
    return AgentRequest(
        command_kind=CommandKind.LOG_CHECK,
        input_path=resolved_target,
        operator_path=None,
        output_path=None,
        test_mode=None,
        bench_mode=None,
        language=language,
        interact=False,
        verbose=verbose,
        stream_output=show_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="ascend-npu-optimize-submit-round",
        prompt=build_log_check_prompt(
            target_path=resolved_target,
            log_check_json_file=output_json,
            pattern_analysis_json_file=pattern_analysis_json,
            agent_name=agent_name,
            language=language,
        ),
        workdir=resolved_target,
        extra_env=extra_env,
        run_id=run_id,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        log_tools=log_tools,
    )


def build_parser(*, prog_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog_name or Path(__file__).name,
        description="Launch Codex log validation and write structured JSON results.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Operator workspace root path containing baseline and opt-round-* directories.",
    )
    parser.add_argument(
        "--output-json",
        default=_LOG_CHECK_JSON_FILENAME,
        help=f"Output JSON filename for check results (default: {_LOG_CHECK_JSON_FILENAME}).",
    )
    parser.add_argument(
        "--pattern-analysis-json",
        default=_PATTERN_ANALYSIS_JSON_FILENAME,
        help=f"Output JSON filename for pattern analysis (default: {_PATTERN_ANALYSIS_JSON_FILENAME}).",
    )
    return parser


def run_log_check(
    *,
    target_path: Path,
    output_json: str = _LOG_CHECK_JSON_FILENAME,
    pattern_analysis_json: str = _PATTERN_ANALYSIS_JSON_FILENAME,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    log_tools: bool = False,
    language: Literal["triton", "tilelang"] = "triton",
) -> int:
    normalized_target = target_path.expanduser().resolve()
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.LOG_CHECK,
        language=language,
    )
    request = build_log_check_request(
        target_path=normalized_target,
        agent_name=agent_name,
        verbose=verbose,
        show_output=show_output,
        output_json=output_json,
        pattern_analysis_json=pattern_analysis_json,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        log_tools=log_tools,
        language=language,
    )
    try:
        runner = create_runner(agent_name)
    except ValueError as exc:
        print(f"[optimize-check] invalid agent: {exc}", file=sys.stderr, flush=True)
        return 2

    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        agent_name,
        normalized_target,
        skill_names=staged_skill_names,
        skill_sources=staged_skill_sources,
    )
    if verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))

    print(
        "[optimize-check] start log check: "
        + (
            f"path={normalized_target.as_posix()}, "
            f"output={output_json}, agent={agent_name}"
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
    finally:
        _write_log_check_trace_summary(request)
        if verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        cleanup_warnings = manager.cleanup(links)
        if cleanup_warnings:
            emit_verbose_lines(sys.stderr, "skills", cleanup_warnings)

    if not result.succeeded:
        if request.stream_output:
            detail = result.stderr.strip() or f"agent execution failed; see show-output log: {show_output_log_path(request)}"
        else:
            detail = result.stderr.strip() or result.stdout.strip() or "agent execution failed"
        print(f"[optimize-check] log check failed: {detail}", file=sys.stderr, flush=True)
        return result.return_code if result.return_code != 0 else 1

    # --- Post-processing: validate JSON, repair if needed, render MD ---
    exit_code = _post_process_log_check_output(
        normalized_target,
        log_check_json_file=output_json,
        pattern_analysis_json_file=pattern_analysis_json,
    )
    return exit_code


def _post_process_log_check_output(
    workspace: Path,
    *,
    log_check_json_file: str,
    pattern_analysis_json_file: str,
) -> int:
    """Validate agent-produced JSON files, repair if needed, and render markdown."""
    log_check_json_path = workspace / log_check_json_file
    pattern_json_path = workspace / pattern_analysis_json_file

    # --- log_check_result.json ---
    log_check_ok = _validate_and_render_log_check(workspace, log_check_json_path)
    if not log_check_ok:
        print(
            "[optimize-check] log check completed but log_check_result.json is missing or invalid",
            file=sys.stderr,
            flush=True,
        )
        return 1

    # --- pattern_analysis.json ---
    _validate_and_render_pattern(workspace, pattern_json_path)

    print(
        "[optimize-check] log check completed: " + log_check_json_path.as_posix(),
        file=sys.stderr,
        flush=True,
    )
    return 0


def _validate_and_render_log_check(workspace: Path, json_path: Path) -> bool:
    """Validate log_check_result.json. On success, render log_check_result.md.
    On failure, attempt repair. Return False if unrecoverable."""
    md_path = workspace / "log_check_result.md"

    if not json_path.is_file():
        print(
            f"[optimize-check] warning: {json_path.name} was not created by agent",
            file=sys.stderr,
            flush=True,
        )
        return False

    raw = json_path.read_text(encoding="utf-8", errors="replace")
    data = _parse_json_with_repair(raw, json_path.name)
    if data is None:
        return False

    errors = validate_log_check_json(data)
    if errors:
        print(
            f"[optimize-check] warning: {json_path.name} validation errors:",
            file=sys.stderr,
            flush=True,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr, flush=True)
        # Still render MD from whatever we have — best-effort
        print(
            f"[optimize-check] rendering {md_path.name} from partial JSON (best-effort)",
            file=sys.stderr,
            flush=True,
        )

    md_content = render_log_check_markdown(data)
    md_path.write_text(md_content, encoding="utf-8")
    return True


def _validate_and_render_pattern(workspace: Path, json_path: Path) -> bool:
    """Validate pattern_analysis.json. On success, render pattern_analysis.md."""
    md_path = workspace / "pattern_analysis.md"

    if not json_path.is_file():
        print(
            f"[optimize-check] warning: {json_path.name} was not created by agent",
            file=sys.stderr,
            flush=True,
        )
        return False

    raw = json_path.read_text(encoding="utf-8", errors="replace")
    data = _parse_json_with_repair(raw, json_path.name)
    if data is None:
        return False

    errors = validate_pattern_analysis_json(data)
    if errors:
        print(
            f"[optimize-check] warning: {json_path.name} validation errors:",
            file=sys.stderr,
            flush=True,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr, flush=True)
        print(
            f"[optimize-check] rendering {md_path.name} from partial JSON (best-effort)",
            file=sys.stderr,
            flush=True,
        )

    md_content = render_pattern_analysis_markdown(data)
    md_path.write_text(md_content, encoding="utf-8")
    return True


def _parse_json_with_repair(raw: str, filename: str) -> dict[str, Any] | None:
    """Parse JSON with repair fallback. Returns dict or None."""
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        print(
            f"[optimize-check] warning: {filename} is not a JSON object",
            file=sys.stderr,
            flush=True,
        )
        return None
    except json.JSONDecodeError as exc:
        print(
            f"[optimize-check] warning: {filename} is invalid JSON: {exc}",
            file=sys.stderr,
            flush=True,
        )
        repaired = repair_json(raw)
        if repaired is not None:
            print(
                f"[optimize-check] repaired {filename} successfully",
                file=sys.stderr,
                flush=True,
            )
            return repaired
        print(
            f"[optimize-check] could not repair {filename}",
            file=sys.stderr,
            flush=True,
        )
        return None


def _write_log_check_trace_summary(request: AgentRequest) -> None:
    if not request.log_tools:
        return
    trace_path = trace_path_from_request(request)
    if trace_path is None:
        return
    warnings = write_tool_trace_summary(
        trace_path=trace_path,
        command_kind=request.command_kind.value,
        show_output_path=show_output_log_path(request),
    )
    if request.verbose and warnings:
        emit_verbose_lines(sys.stderr, "trace", warnings)


def main(argv: list[str] | None = None, *, prog_name: str | None = None) -> int:
    parser = build_parser(prog_name=prog_name)
    args = parser.parse_args(argv)
    return run_log_check(
        target_path=Path(args.path),
        output_json=str(args.output_json),
        pattern_analysis_json=str(args.pattern_analysis_json),
        verbose=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
