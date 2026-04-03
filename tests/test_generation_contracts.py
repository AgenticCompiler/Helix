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
        content = _read("skills/test-gen/SKILL.md")
        self.assertIn("# test-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# api-kind:", content)
        self.assertIn("# kernel:", content)
        self.assertIn("accept only `--operator-file`", content)
        self.assertNotIn("must accept `--operator-file` and `--api-name`", content)

    def test_bench_gen_skill_requires_header_metadata_and_no_runtime_api_flag(self) -> None:
        content = _read("skills/bench-gen/SKILL.md")
        self.assertIn("# bench-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# api-kind:", content)
        self.assertIn("# kernel:", content)
        self.assertIn("accept only `--operator-file` at runtime for standalone mode", content)
        self.assertNotIn("must accept `--operator-file` and `--api-name`", content)

    def test_generation_skills_support_entrypoint_kinds(self) -> None:
        for relative_path in ("skills/test-gen/SKILL.md", "skills/bench-gen/SKILL.md"):
            content = _read(relative_path)
            with self.subTest(path=relative_path):
                self.assertIn("triton-wrapper", content)
                self.assertIn("torch-function", content)
                self.assertIn("torch-module", content)
                self.assertIn("public entrypoint", content)
                self.assertIn("Do not", content)
                self.assertIn("constructor", content)

    def test_generation_and_optimize_skills_do_not_reference_removed_run_skills(self) -> None:
        self.assertNotIn("skill `test-run`", _read("skills/test-gen/SKILL.md"))
        self.assertNotIn("`bench-run`", _read("skills/bench-gen/SKILL.md"))
        optimize = _read("skills/optimize/SKILL.md")
        self.assertNotIn("run-test --input", optimize)
        self.assertNotIn("run-bench --input", optimize)

    def test_generation_skills_include_explicit_run_command_examples(self) -> None:
        test_gen = _read("skills/test-gen/SKILL.md")
        self.assertIn("## Validation Commands", test_gen)
        self.assertIn("Use the operator-eval skill to execute generated test cases.", test_gen)
        self.assertIn("python3 ../operator-eval/scripts/run-command.py run-test --test-file", test_gen)
        self.assertIn("Do not run `compare-result` during test generation.", test_gen)
        self.assertNotIn("run `compare-result` after `run-test` succeeds", test_gen)

        bench_gen = _read("skills/bench-gen/SKILL.md")
        self.assertIn("## Validation Commands", bench_gen)
        self.assertIn("Use the operator-eval skill to execute generated benchmark cases.", bench_gen)
        self.assertIn("python3 ../operator-eval/scripts/run-command.py run-bench --bench-file", bench_gen)

    def test_optimize_skill_includes_remote_command_examples(self) -> None:
        optimize = _read("skills/optimize/SKILL.md")
        self.assertIn(
            "Use the bundled helper script at [`../operator-eval/scripts/run-command.py`](../operator-eval/scripts/run-command.py) for generation, validation, profiling, and comparison commands; if the outer optimize task is remote-aware, carry the same remote flags through those commands.",
            optimize,
        )
        self.assertIn(
            "Generate missing tests or benchmarks through `../operator-eval/scripts/run-command.py` before starting any optimization round.",
            optimize,
        )
        self.assertIn("ascend-npu-operator-profiler", optimize)

    def test_optimize_skill_allows_non_pattern_optimization_knowledge(self) -> None:
        optimize = _read("skills/optimize/SKILL.md")
        self.assertIn("Pattern references are helpful guidance, not the only allowed source of ideas.", optimize)
        self.assertIn("If your own Triton, Ascend NPU, or kernel-optimization knowledge suggests a stronger direction", optimize)
        self.assertIn("You do not need an existing pattern file to justify every optimization round.", optimize)

    def test_optimize_skill_records_learned_lessons(self) -> None:
        optimize = _read("skills/optimize/SKILL.md")
        self.assertIn("learned_lessons.md", optimize)
        self.assertIn("record learned lessons whenever you discover reusable knowledge", optimize)
        self.assertIn("compiler error repairs", optimize)
        self.assertIn("profile-guided optimization lessons", optimize)

    def test_profiler_skill_documents_profile_bench_mode_contracts(self) -> None:
        profiler = _read("skills/ascend-npu-operator-profiler/SKILL.md")
        self.assertIn("../operator-eval/scripts/run-command.py profile-bench", profiler)
        self.assertIn("standalone", profiler)
        self.assertIn("msprof", profiler)
        self.assertIn("must not receive `--bench` or `--num-bench`", profiler)
        self.assertIn("first query `--num-bench`", profiler)
        self.assertIn("profile one selected `--bench <N>` case", profiler)

    def test_test_generation_specs_use_only_operator_file_cli(self) -> None:
        standalone = _read("skills/test-gen/references/test-standalone-spec.md")
        differential = _read("skills/test-gen/references/test-differential-spec.md")

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
        standalone = _read("skills/bench-gen/references/bench-standalone-spec.md")
        msprof = _read("skills/bench-gen/references/bench-msprof-spec.md")

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
        test_spec = _read("skills/test-gen/references/test-standalone-spec.md")
        bench_spec = _read("skills/bench-gen/references/bench-standalone-spec.md")

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
