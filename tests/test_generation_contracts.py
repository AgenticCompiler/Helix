import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class GenerationContractTests(unittest.TestCase):
    def test_test_gen_skill_requires_header_metadata_and_no_runtime_api_flag(self) -> None:
        content = _read("skills/test-gen/SKILL.md")
        self.assertIn("# test-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# kernel:", content)
        self.assertIn("accept only `--operator-file`", content)
        self.assertNotIn("must accept `--operator-file` and `--api-name`", content)

    def test_bench_gen_skill_requires_header_metadata_and_no_runtime_api_flag(self) -> None:
        content = _read("skills/bench-gen/SKILL.md")
        self.assertIn("# bench-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# kernel:", content)
        self.assertIn("accept only `--operator-file` at runtime for standalone mode", content)
        self.assertNotIn("must accept `--operator-file` and `--api-name`", content)

    def test_generation_and_optimize_skills_do_not_reference_removed_run_skills(self) -> None:
        self.assertNotIn("skill `test-run`", _read("skills/test-gen/SKILL.md"))
        self.assertNotIn("`bench-run`", _read("skills/bench-gen/SKILL.md"))
        optimize = _read("skills/optimize/SKILL.md")
        self.assertNotIn("run-test --input", optimize)
        self.assertNotIn("run-bench --input", optimize)

    def test_generation_skills_include_explicit_run_command_examples(self) -> None:
        test_gen = _read("skills/test-gen/SKILL.md")
        self.assertIn("## Validation Commands", test_gen)
        self.assertIn("python3 ../scripts/run-command.py run-test --test-file", test_gen)
        self.assertIn("python3 differential_test_<operator>.py --operator-file", test_gen)
        self.assertIn("--remote user@host:2222", test_gen)
        self.assertIn("--remote-workdir /tmp/triton-agent", test_gen)

        bench_gen = _read("skills/bench-gen/SKILL.md")
        self.assertIn("## Validation Commands", bench_gen)
        self.assertIn("python3 ../scripts/run-command.py run-bench --bench-file", bench_gen)
        self.assertIn("python3 bench_<operator>.py --num-bench", bench_gen)
        self.assertIn("--remote user@host:2222", bench_gen)
        self.assertIn("--remote-workdir /tmp/triton-agent", bench_gen)

    def test_optimize_skill_includes_remote_command_examples(self) -> None:
        optimize = _read("skills/optimize/SKILL.md")
        self.assertIn("gen-test --input <operator.py> --test-mode <mode> --remote", optimize)
        self.assertIn("gen-bench --input <operator.py> --bench-mode <mode> --remote", optimize)
        self.assertIn("run-test --test-file <test.py> --operator-file <candidate.py>", optimize)
        self.assertIn("run-bench --bench-file <bench.py> --operator-file <candidate.py>", optimize)
        self.assertIn("--remote-workdir /tmp/triton-agent", optimize)

    def test_test_generation_specs_use_only_operator_file_cli(self) -> None:
        standalone = _read("skills/test-gen/references/test-standalone-spec.md")
        differential = _read("skills/test-gen/references/test-differential-spec.md")

        for content in (standalone, differential):
            with self.subTest(spec=content[:40]):
                self.assertIn("# api-name: <name>", content)
                self.assertIn("# kernel: <name>", content)
                self.assertIn("# api-name: <resolved_wrapper_api>", content)
                self.assertIn("# kernel: <resolved_kernel_name>", content)
                self.assertNotIn("| `--api-name <name>` | yes |", content)
                self.assertIn("Parses `--operator-file`", content)

    def test_benchmark_generation_specs_use_header_metadata_and_no_runtime_api_flag(self) -> None:
        standalone = _read("skills/bench-gen/references/bench-standalone-spec.md")
        msprof = _read("skills/bench-gen/references/bench-msprof-spec.md")

        self.assertIn("# bench-mode: standalone", standalone)
        self.assertIn("# api-name: <resolved_wrapper_api>", standalone)
        self.assertIn("# kernel: <resolved_kernel_name>", standalone)
        self.assertNotIn("| `--api-name <name>` | yes |", standalone)
        self.assertIn("parses `--operator-file`", standalone.lower())
        self.assertIn("def load_operator_api(operator_file: str, api_name: str):", standalone)
        self.assertIn("def run_bench(operator_api):", standalone)
        self.assertIn("triton.backends.ascend.testing.do_bench_npu", standalone)
        self.assertIn('print(f"latency-{case_id}: {latency}")', standalone)

        self.assertIn("# bench-mode: msprof", msprof)
        self.assertIn("# api-name: <resolved_wrapper_api>", msprof)
        self.assertIn("# kernel: <resolved_kernel_name>", msprof)
        self.assertNotIn("--api-name <api-name>", msprof)
        self.assertIn("If `--bench N` is provided, then `--operator-file` is required.", msprof)

    def test_workspace_examples_follow_metadata_header_contract(self) -> None:
        test_example = _read("workspace/matmul/test_matmul.py")
        bench_example = _read("workspace/matmul/bench_matmul.py")

        self.assertIn("# test-mode:", test_example)
        self.assertIn("# api-name:", test_example)
        self.assertIn("# kernel:", test_example)
        self.assertIn('parser.add_argument("--operator-file", required=True)', test_example)
        self.assertNotIn('parser.add_argument("--api-name"', test_example)

        self.assertIn("# bench-mode:", bench_example)
        self.assertIn("# api-name:", bench_example)
        self.assertIn("# kernel:", bench_example)
        self.assertIn('parser.add_argument("--operator-file"', bench_example)
        self.assertNotIn('parser.add_argument("--api-name"', bench_example)
        if "# bench-mode: msprof" in bench_example:
            self.assertIn('parser.add_argument("--bench", type=int)', bench_example)
            self.assertIn('parser.add_argument("--num-bench", action="store_true")', bench_example)
        if "# bench-mode: standalone" in bench_example:
            self.assertIn("triton.backends.ascend.testing.do_bench_npu", bench_example)
            self.assertIn('print(f"latency-', bench_example)


if __name__ == "__main__":
    unittest.main()
