import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class GenerationContractTests(unittest.TestCase):
    def test_pyright_configuration_keeps_tests_basic_while_src_is_strict(self) -> None:
        content = _read("pyproject.toml")
        self.assertIn('typeCheckingMode = "basic"', content)
        self.assertIn('strict = ["src"]', content)

    def test_test_gen_skill_requires_header_metadata_and_no_runtime_api_flag(self) -> None:
        content = _read("skills/triton-npu-gen-test/SKILL.md")
        self.assertIn("# test-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# api-kind:", content)
        self.assertIn("# kernel:", content)
        self.assertIn("accept only `--operator-file`", content)
        self.assertNotIn("must accept `--operator-file` and `--api-name`", content)

    def test_bench_gen_skill_requires_header_metadata_and_no_runtime_api_flag(self) -> None:
        content = _read("skills/triton-npu-gen-bench/SKILL.md")
        self.assertIn("# bench-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# api-kind:", content)
        self.assertIn("# kernel:", content)
        self.assertIn("accept only `--operator-file` at runtime for standalone mode", content)
        self.assertNotIn("must accept `--operator-file` and `--api-name`", content)

    def test_generation_skills_support_entrypoint_kinds(self) -> None:
        for relative_path in ("skills/triton-npu-gen-test/SKILL.md", "skills/triton-npu-gen-bench/SKILL.md"):
            content = _read(relative_path)
            with self.subTest(path=relative_path):
                self.assertIn("triton-wrapper", content)
                self.assertIn("torch-function", content)
                self.assertIn("torch-module", content)
                self.assertIn("public entrypoint", content)
                self.assertIn("Do not", content)
                self.assertIn("constructor", content)

    def test_generation_and_optimize_skills_do_not_reference_removed_run_skills(self) -> None:
        self.assertNotIn("skill `test-run`", _read("skills/triton-npu-gen-test/SKILL.md"))
        self.assertNotIn("`bench-run`", _read("skills/triton-npu-gen-bench/SKILL.md"))
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertNotIn("run-test --input", optimize)
        self.assertNotIn("run-bench --input", optimize)

    def test_generation_skills_include_explicit_run_command_examples(self) -> None:
        test_gen = _read("skills/triton-npu-gen-test/SKILL.md")
        self.assertIn("## Validation Commands", test_gen)
        self.assertIn("Use the triton-npu-run-eval skill to execute generated test cases.", test_gen)
        self.assertIn("python3 ../triton-npu-run-eval/scripts/run-command.py run-test --test-file", test_gen)
        self.assertIn("Do not run `compare-result` during test generation.", test_gen)
        self.assertNotIn("run `compare-result` after `run-test` succeeds", test_gen)

        bench_gen = _read("skills/triton-npu-gen-bench/SKILL.md")
        self.assertIn("## Validation Commands", bench_gen)
        self.assertIn("Use the triton-npu-run-eval skill to execute generated benchmark cases.", bench_gen)
        self.assertIn("python3 ../triton-npu-run-eval/scripts/run-command.py run-bench --bench-file", bench_gen)

    def test_eval_gen_skill_documents_direct_operator_repair_and_remote_validation(self) -> None:
        eval_gen = _read("skills/triton-npu-gen-eval-suite/SKILL.md")
        self.assertIn("repair the original operator file", eval_gen)
        self.assertIn("triton-npu-gen-test", eval_gen)
        self.assertIn("triton-npu-gen-bench", eval_gen)
        self.assertIn("triton-npu-run-eval", eval_gen)
        self.assertIn("Use the `triton-npu-run-eval` skill for correctness validation", eval_gen)
        self.assertIn("Use the `triton-npu-run-eval` skill for benchmark validation", eval_gen)
        self.assertIn("carry the same remote flags", eval_gen)
        self.assertIn("Do not", eval_gen)
        self.assertIn("opt-round", eval_gen)

    def test_optimize_skill_includes_remote_command_examples(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertIn(
            "Use the bundled helper script at [`../triton-npu-run-eval/scripts/run-command.py`](../triton-npu-run-eval/scripts/run-command.py) for generation, validation, profiling, and comparison commands; if the outer optimize task is remote-aware, carry the same remote flags through those commands.",
            optimize,
        )
        self.assertIn(
            "Generate missing tests or benchmarks through `../triton-npu-run-eval/scripts/run-command.py` before starting any optimization round.",
            optimize,
        )
        self.assertIn("triton-npu-profile-operator", optimize)
        self.assertIn("triton-npu-analyze-round-performance", optimize)
        self.assertIn("`opt-round-N/perf-analysis.md`", optimize)

    def test_optimize_artifacts_reference_documents_state_declared_paths(self) -> None:
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")
        self.assertIn("Treat these state fields as the authoritative artifact references for baseline validation:", artifacts)
        self.assertIn("`baseline_operator`", artifacts)
        self.assertIn("`perf_artifact`", artifacts)
        self.assertIn("Treat these round-state fields as the authoritative artifact references for round validation:", artifacts)
        self.assertIn("`summary_path`", artifacts)
        self.assertIn("`perf_analysis_path` when present", artifacts)
        self.assertIn("`profile_dir` when present", artifacts)
        self.assertIn("`ir_dir` when present", artifacts)

    def test_optimize_skill_documents_round_local_ir_commands(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertIn("python3 ../triton-npu-analyze-ir/scripts/capture_ir.py", optimize)
        self.assertIn("--ir-dir opt-round-N/ir", optimize)
        self.assertIn("python3 ../triton-npu-analyze-ir/scripts/inspect_ir.py", optimize)

    def test_round_performance_skill_describes_layered_profiler_and_binary_analysis(self) -> None:
        content = _read("skills/triton-npu-analyze-round-performance/SKILL.md")
        profiling_ref = _read(
            "skills/triton-npu-analyze-round-performance/references/ascend-npu-profiling-analysis.md"
        )
        architecture_ref = _read(
            "skills/triton-npu-analyze-round-performance/references/ascend-npu-architecture-notes.md"
        )
        self.assertIn("ascend-npu-optimization-guidance.md", content)
        self.assertIn("ascend-npu-profiling-analysis.md", content)
        self.assertIn("ascend-npu-architecture-notes.md", content)
        self.assertIn("Read the references in this order", content)
        self.assertIn("1. profiling analysis", content)
        self.assertIn("2. optimization guidance", content)
        self.assertIn("3. architecture notes", content)
        self.assertIn("profiler-first layered analysis", content)
        self.assertIn("IR as explanation and attribution", content)
        self.assertIn("two complementary analysis paths", content)
        self.assertIn("potential optimization points", content)
        self.assertIn("## Binary Signals", content)
        self.assertIn("## Diagnosis", content)
        self.assertIn("Operator Type Fit", content)
        self.assertIn("Compute vs Memory Bound", content)
        self.assertIn("Concurrency And Scheduling Bottlenecks", content)
        self.assertIn("`op_statistic`", profiling_ref)
        self.assertIn("`op_summary`", profiling_ref)
        self.assertIn("`task_time`", profiling_ref)
        self.assertIn("`api_statistic`", profiling_ref)
        self.assertIn("`msprof` JSON", profiling_ref)
        self.assertIn("`.bin`", profiling_ref)
        self.assertIn("A3", architecture_ref)
        self.assertIn("A5", architecture_ref)
        self.assertIn("L0C", architecture_ref)

    def test_compiler_source_analysis_skill_documents_read_only_cli_provisioned_source(self) -> None:
        content = _read("skills/triton-npu-analyze-compiler-source/SKILL.md")

        self.assertIn("CLI-provided local source path", content)
        self.assertIn("Do not run `git clone`, `git fetch`, or `git pull`", content)
        self.assertIn("Treat the compiler source checkout as read-only", content)
        self.assertIn("AscendNPU-IR", content)
        self.assertIn("opt-round-N/compiler-analysis.md", content)
        self.assertIn("Source Files Inspected", content)
        self.assertIn("Confidence And Evidence Gaps", content)
        self.assertIn("version mismatch", content)
        self.assertIn("Detailed source indexing is deferred", content)

    def test_optimize_skills_document_compiler_source_escalation(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        workflow = _read("skills/triton-npu-optimize/references/workflow.md")
        round_analysis = _read("skills/triton-npu-analyze-round-performance/SKILL.md")

        self.assertIn("triton-npu-analyze-compiler-source", optimize)
        self.assertIn("compiler source analysis is enabled", optimize)
        self.assertIn("after profiler and IR evidence", workflow)
        self.assertIn("opt-round-N/compiler-analysis.md", workflow)
        self.assertIn("compiler source analysis is enabled", round_analysis)

    def test_readme_documents_compiler_source_analysis_options(self) -> None:
        content = _read("README.md")

        self.assertIn("--enable-compiler-source-analysis", content)
        self.assertIn("--compiler-source-path <path>", content)
        self.assertIn("~/.triton-agent/compiler-sources/AscendNPU-IR/", content)
        self.assertIn("read-only", content)
        self.assertIn("escalation", content)

    def test_optimize_skill_allows_non_pattern_optimization_knowledge(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertIn("Pattern references are helpful guidance, not the only allowed source of ideas.", optimize)
        self.assertIn("If your own Triton, Ascend NPU, or kernel-optimization knowledge suggests a stronger direction", optimize)
        self.assertIn("You do not need an existing pattern file to justify every optimization round.", optimize)

    def test_optimize_skill_records_learned_lessons(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertIn("learned_lessons.md", optimize)
        self.assertIn("strict reusable optimization-knowledge distillation log", optimize)
        self.assertIn("passes all admission criteria", optimize)
        self.assertIn("supported by correctness, benchmark, profiler, IR, or compiler-error evidence", optimize)
        self.assertIn("states where it applies or what limits it", optimize)
        self.assertIn("could plausibly be promoted into an optimize skill", optimize)
        self.assertIn("Do not use `learned_lessons.md` for round narrative", optimize)

    def test_optimize_artifacts_document_strict_learned_lessons_boundary(self) -> None:
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")
        self.assertIn("strict reusable optimization-knowledge log", artifacts)
        self.assertIn("Only add an entry when it is evidence-backed", artifacts)
        self.assertIn("portable to related Triton Ascend NPU operators", artifacts)
        self.assertIn("Round-local command failures", artifacts)
        self.assertIn("shape-specific details", artifacts)
        self.assertIn("belong in `attempts.md`, `summary.md`, or `opt-note.md`", artifacts)

    def test_repair_guide_skill_owns_novel_fix_logging(self) -> None:
        repair_guide = _read("skills/triton-npu-repair-guide/SKILL.md")
        self.assertIn("append a short entry to [output.md](output.md)", repair_guide)
        self.assertIn("Append-Only Repair Log", repair_guide)
        self.assertIn("through the `triton-npu-run-eval` skill", repair_guide)
        self.assertFalse((REPO_ROOT / "skills" / "triton-npu-log-repair").exists())

    def test_cross_skill_subcommands_name_owning_skills(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertIn("use the `triton-npu-run-eval` skill to run `compare-perf`", optimize)
        self.assertIn("the `triton-npu-run-eval` skill's `compare-perf` flow", optimize)

    def test_profiler_skill_documents_profile_bench_mode_contracts(self) -> None:
        profiler = _read("skills/triton-npu-profile-operator/SKILL.md")
        self.assertIn("../triton-npu-run-eval/scripts/run-command.py profile-bench", profiler)
        self.assertIn("standalone", profiler)
        self.assertIn("msprof", profiler)
        self.assertIn("must not receive `--bench` or `--num-bench`", profiler)
        self.assertIn("first query `--num-bench`", profiler)
        self.assertIn("profile one selected `--bench <N>` case", profiler)

    def test_test_generation_specs_use_only_operator_file_cli(self) -> None:
        standalone = _read("skills/triton-npu-gen-test/references/test-standalone-spec.md")
        differential = _read("skills/triton-npu-gen-test/references/test-differential-spec.md")

        for content in (standalone, differential):
            with self.subTest(spec=content[:40]):
                self.assertIn("# api-name: <name>", content)
                self.assertIn("# api-kind: <triton-wrapper|torch-function|torch-module>", content)
                self.assertIn("# kernel: <name>", content)
                self.assertIn("# api-name: <resolved_entrypoint>", content)
                self.assertIn("# api-kind: <resolved_api_kind>", content)
                self.assertIn("# kernel: <resolved_kernel_name>", content)
                self.assertNotIn("| `--api-name <name>` | yes |", content)
                self.assertIn("Parses `--operator-file`", content)
                self.assertIn("triton-wrapper", content)
                self.assertIn("torch-function", content)
                self.assertIn("torch-module", content)

    def test_benchmark_generation_specs_use_header_metadata_and_no_runtime_api_flag(self) -> None:
        standalone = _read("skills/triton-npu-gen-bench/references/bench-standalone-spec.md")
        msprof = _read("skills/triton-npu-gen-bench/references/bench-msprof-spec.md")

        self.assertIn("# bench-mode: standalone", standalone)
        self.assertIn("# api-name: <resolved_entrypoint>", standalone)
        self.assertIn("# api-kind: <resolved_api_kind>", standalone)
        self.assertIn("# kernel: <resolved_kernel_name>", standalone)
        self.assertNotIn("| `--api-name <name>` | yes |", standalone)
        self.assertIn("parses `--operator-file`", standalone.lower())
        self.assertIn("#### 3.1 `triton-wrapper`", standalone)
        self.assertIn("#### 3.2 `torch-function`", standalone)
        self.assertIn("#### 3.3 `torch-module`", standalone)
        self.assertIn("def load_operator_api(operator_file: str, api_name: str):", standalone)
        self.assertIn("def run_bench(operator_api):", standalone)
        self.assertIn("triton.backends.ascend.testing.do_bench_npu", standalone)
        self.assertIn('print(f"latency-{case_id}: {latency}")', standalone)
        self.assertIn("torch-module", standalone)
        self.assertIn("constructor arguments", standalone)

        self.assertIn("# bench-mode: msprof", msprof)
        self.assertIn("# api-name: <resolved_entrypoint>", msprof)
        self.assertIn("# api-kind: <resolved_api_kind>", msprof)
        self.assertIn("# kernel: <resolved_kernel_name>", msprof)
        self.assertNotIn("--api-name <api-name>", msprof)
        self.assertIn("If `--bench N` is provided, then `--operator-file` is required.", msprof)
        self.assertIn("torch-function", msprof)

    def test_contracts_do_not_depend_on_workspace_placeholder_examples(self) -> None:
        test_spec = _read("skills/triton-npu-gen-test/references/test-standalone-spec.md")
        bench_spec = _read("skills/triton-npu-gen-bench/references/bench-standalone-spec.md")

        self.assertIn("# test-mode:", test_spec)
        self.assertIn("# api-name:", test_spec)
        self.assertIn("# api-kind:", test_spec)
        self.assertIn("# kernel:", test_spec)
        self.assertIn('parser.add_argument("--operator-file", required=True)', test_spec)
        self.assertNotIn('parser.add_argument("--api-name"', test_spec)

        self.assertIn("# bench-mode:", bench_spec)
        self.assertIn("# api-name:", bench_spec)
        self.assertIn("# api-kind:", bench_spec)
        self.assertIn("# kernel:", bench_spec)
        self.assertIn('parser.add_argument("--operator-file"', bench_spec)
        self.assertNotIn('parser.add_argument("--api-name"', bench_spec)
        self.assertIn("triton.backends.ascend.testing.do_bench_npu", bench_spec)
        self.assertIn('print(f"latency-', bench_spec)


if __name__ == "__main__":
    unittest.main()
