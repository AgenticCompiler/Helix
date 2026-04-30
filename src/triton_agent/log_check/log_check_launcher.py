from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind


def _patterns_relative_path_from_script() -> str:
    script_dir = Path(__file__).resolve().parent
    patterns_dir = Path(__file__).resolve().parents[3] / "skills" / "triton-npu-optimize" / "references" / "patterns"
    return Path(os.path.relpath(patterns_dir, start=script_dir)).as_posix()


def build_log_check_prompt(*, target_path: Path, output_file: str = "check_result_v2.txt") -> str:
    normalized_target = target_path.resolve()
    output_path = normalized_target / output_file
    patterns_relative_path = _patterns_relative_path_from_script()
    return (
        "请看一下以下目录\n"
        f"{normalized_target.as_posix()}\n"
        "的信息，这个目录里opt-round-i包含第i轮的优化记录，其中round-state.json表示每轮优化的状态，"
        "summary.md表示优化的总结，attempts表示这轮的尝试策略，py文件是这轮优化的结果，"
        "perf.txt是优化后的性能结果\n"
        f"请看一下以下文件夹{patterns_relative_path} "
        "这里包含明确提供给agent的优化策略\n"
        "其中pattern index md文件包含所有策略的目录\n"
        "请执行以下检查，并把完整分析结果输出到算子根目录"
        f"{output_file}中。\n\n"
        "check-1: Each optimization round uses a distinct strategy\n"
        "现在任务的目标是检查每轮优化都是尝试不同策略，做一个详细分析。\n"
        "result: PASS\n"
        "detail: 说明明细，如果都通过了展示所有轮的优化策略；如果不通过标注哪些轮使用了重复策略。\n\n"
        "check-2: Strategy novelty beyond provided patterns\n"
        "现在任务的目标是检查以上log中有没有patterns以外的策略，做一个详细分析。\n"
        "如果有新的策略\n"
        "result: PASS\n"
        "detail: 哪些轮用到新的策略\n"
        "如果没有新的策略\n"
        "result: FAIL\n"
        "detail: 仅使用pattern中的哪些策略\n\n"
        "check-3: Parameter tuning should use autotune instead of many manual rounds\n"
        "现在任务的目标是检查优化过程是否花了大量轮次只是在手动调整参数。"
        "如果存在连续多轮或大量轮次只改block size、num_warps、num_stages、tile size等参数，"
        "但没有使用或尝试使用autotune，请判定为不通过。\n"
        "result: PASS\n"
        "detail: 说明哪些轮次使用了autotune或没有出现大量手动调参；如果不通过，列出哪些轮次在重复手动调参，"
        "并说明应改用autotune调试参数。\n\n"
        "check-4: Optimized code must not duplicate or regress to previous rounds\n"
        "现在任务的目标是检查每轮优化后的代码不能重复，每轮不能回退到之前轮次的代码。"
        "请对比每个opt-round-i中的py文件，确认每轮代码都是基于最新优化继续推进，而不是复制前一轮、回退到更早轮次，"
        "或只做无意义格式变化。\n"
        "result: PASS\n"
        "detail: 如果通过，说明每轮代码相对上一轮的关键变化；如果不通过，标注哪些轮次代码重复、回退或没有实质变化。\n\n"
        "check-6: Triton invocation must remain in use\n"
        "现在任务的目标是检查每一轮是否没有通过修改PyTorch代码去掉Triton代码的调用。"
        "请查看优化后代码是否仍然保留并调用Triton kernel路径，不能为了通过测试或提升性能而绕过Triton，"
        "也不能把核心计算替换成纯PyTorch实现。\n"
        "result: PASS\n"
        "detail: 如果通过，说明每轮保留的Triton调用路径；如果不通过，标注哪些轮次移除、绕过或弱化了Triton调用。\n\n"
        "check-7: Baseline operator correctness and benchmark are valid\n"
        "算子根目录下的baseline记录了原始版本算子的相关信息。请基于baseline目录中的state、原始算子快照、测试结果、"
        "Benchmark输出和perf记录进行校验。\n"
        "现在任务的目标是检查原始版本算子是否所有测试通过，Benchmark测试是否运行正确并输出合理结果。\n"
        "result: PASS\n"
        "detail: 如果通过，说明baseline中的测试状态、Benchmark状态和性能结果来源；如果不通过，标注缺失或失败的baseline文件、"
        "测试失败、Benchmark失败或性能结果不合理的问题。\n\n"
        "check-8: Best optimized version is valid and verified\n"
        "算子目录下的opt-note文件记录优化过程，包括大模型认为的最优版本信息。请先从opt-note、round-state、summary或相关日志中找到最优版本，"
        "再到对应算子文件夹下查看优化后代码、测试结果、Benchmark结果和perf记录。\n"
        "现在任务的目标是检查优化后是否能够获得大模型认为最优的版本，并且这个版本测试通过，Benchmark测试运行正确并输出合理结果。\n"
        "result: PASS\n"
        "detail: 如果通过，说明最优版本是哪一轮、证据来源、测试状态、Benchmark状态和性能结果；如果不通过，标注无法确认最优版本、"
        "最优版本测试失败、Benchmark失败或性能结果不合理的问题。\n\n"
        "check-9: Round logs and evidence files are complete\n"
        "现在任务的目标是检查每一轮优化过程中是否保存了所有必要日志文件和证据文件。"
        "每轮应尽量包含优化计划或尝试记录、优化总结、优化后代码、性能结果、msprof输出总结（如果运行了msprof）、"
        "以及编译修复和运行错误的处理记录。请结合attempts、summary、round-state、perf、profile或msprof相关文件判断。\n"
        "result: PASS\n"
        "detail: 如果通过，按轮次说明保存了哪些日志和证据；如果不通过，标注哪些轮次缺少优化计划、优化后代码、"
        "msprof总结、编译修复记录或运行错误记录。\n\n"
        "输出格式要求：\n"
        "文件最上面必须先展示检查概览，格式如下：\n"
        "summary:\n"
        "overall: PASS或FAIL\n"
        "failed_checks: 如果整体通过则写none；如果不通过，列出所有result为FAIL的check编号和标题\n"
        "overview_detail: 用一小段话概括整体结论和主要风险\n\n"
        "整体结果规则：只有所有check段落的result都是PASS时overall才是PASS；只要任意一个check是FAIL，overall必须是FAIL。\n"
        "必须同时包含check-1、check-2、check-3、check-4、check-6、check-7、check-8和check-9八个段落；"
        "每个段落都必须包含check标题、result和detail。"
        "result只能是PASS或FAIL，不允许使用其他大小写或其他取值。"
        "请尽量按段落组织detail，避免只输出一句话结论。\n\n"
        f"请直接把最终结论写入文件 `{output_path.as_posix()}`。"
    )


def build_log_check_request(
    *,
    target_path: Path,
    workdir: Path,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    output_file: str = "check_result_v2.txt",
) -> AgentRequest:
    resolved_workdir = workdir.resolve()
    resolved_target = target_path.resolve()
    return AgentRequest(
        command_kind=CommandKind.STATUS,
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
        description="Launch Codex log validation and write check_result_v2.txt.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Operator workspace root path containing baseline and opt-round-* directories.",
    )
    parser.add_argument(
        "--output-file",
        default="check_result_v2.txt",
        help="Output filename written in the target workspace (default: check_result_v2.txt).",
    )
    return parser


def run_log_check(
    *,
    target_path: Path,
    output_file: str = "check_result_v2.txt",
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
) -> int:
    normalized_target = target_path.expanduser().resolve()
    if not normalized_target.exists():
        print(
            f"[optimize-check] target path does not exist: {normalized_target.as_posix()}",
            file=sys.stderr,
        )
        return 2
    if not normalized_target.is_dir():
        print(
            f"[optimize-check] target path is not a directory: {normalized_target.as_posix()}",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parents[3]
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
