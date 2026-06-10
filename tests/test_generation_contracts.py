import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


class GenerationContractTests(unittest.TestCase):
    def test_pyright_configuration_keeps_tests_basic_while_src_is_strict(self) -> None:
        content = _read("pyproject.toml")
        self.assertIn('include = ["src", "tests", "skills/*/scripts"]', content)
        self.assertIn('typeCheckingMode = "basic"', content)
        self.assertIn('strict = ["src"]', content)
        self.assertNotIn('"skills",', content)

    def test_skill_script_pyright_wrapper_is_documented_in_agents(self) -> None:
        agents = _read("AGENTS.md")
        wrapper = _read("scripts/run-skill-script-pyright.sh")
        self.assertIn("scripts/run-skill-script-pyright.sh", agents)
        self.assertIn(
            "bash scripts/run-skill-script-pyright.sh skills/path/to/script.py",
            agents,
        )
        self.assertIn("UV_CACHE_DIR", wrapper)
        self.assertIn("uv run pyright", wrapper)
        self.assertIn('typeCheckingMode = "strict"', wrapper)

    def test_gitcode_pr_skill_uses_official_api_script(self) -> None:
        skill = _read(".codex/skills/managing-gitcode-prs/SKILL.md")
        reference = _read(".codex/skills/managing-gitcode-prs/references/pr-command-reference.md")
        script = _read(".codex/skills/managing-gitcode-prs/scripts/gitcode_pr_api.py")

        self.assertIn("scripts/gitcode_pr_api.py", skill)
        self.assertIn("midwinter1993/triton-agent", skill)
        self.assertIn("scripts/gitcode_pr_api.py", reference)
        self.assertIn("Authorization", script)
        self.assertIn("Bearer", script)
        self.assertIn("GC_TOKEN", script)
        self.assertIn("https://gitcode.com/api/v5/repos", script)
        self.assertIn("--prune-source-branch", script)
        self.assertNotIn("run-gc-pr.sh", skill)
        self.assertNotIn("uv tool run --from", reference)

    def test_gitcode_pr_skill_prunes_source_branch_with_official_api_field(self) -> None:
        script = (
            REPO_ROOT
            / ".codex"
            / "skills"
            / "managing-gitcode-prs"
            / "scripts"
            / "gitcode_pr_api.py"
        )
        spec = importlib.util.spec_from_file_location("gitcode_pr_api_test", script)
        if spec is None or spec.loader is None:
            self.fail(f"Unable to load module spec for {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        calls: list[tuple[str, str, Optional[dict[str, object]]]] = []

        def fake_request_json(
            token: str,
            url: str,
            *,
            method: str = "GET",
            payload: Optional[dict[str, object]] = None,
            disable_proxy: bool = False,
        ) -> dict[str, object]:
            calls.append((method, url, payload))
            if method == "POST":
                return {
                    "number": 17,
                    "title": "Feature",
                    "state": "open",
                    "source_branch": "feature-branch",
                    "target_branch": "main",
                    "created_at": "2026-05-06T00:00:00Z",
                    "updated_at": "2026-05-06T00:00:00Z",
                }
            return {
                "number": 17,
                "title": "Feature",
                "state": "open",
                "source_branch": "feature-branch",
                "target_branch": "main",
                "created_at": "2026-05-06T00:00:00Z",
                "updated_at": "2026-05-06T00:00:00Z",
            }

        args = SimpleNamespace(
            repo="midwinter1993/triton-agent",
            title="Feature",
            body=None,
            head="feature-branch",
            base="main",
            fill=False,
            draft=False,
            prune_source_branch=True,
            json=True,
        )

        with (
            patch.object(module, "require_token", return_value="token"),
            patch.object(module, "request_json", side_effect=fake_request_json),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(module.command_create(args), 0)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(
            calls[0][2],
            {"title": "Feature", "head": "feature-branch", "base": "main"},
        )
        self.assertEqual(calls[1][0], "PATCH")
        self.assertEqual(calls[1][2], {"force_remove_source_branch": True})

    def test_test_gen_skill_emphasizes_spec_and_shared_contracts(self) -> None:
        content = _read("skills/triton-npu-gen-test/SKILL.md")
        self.assertIn("# test-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# api-kind:", content)
        self.assertIn("# kernels:", content)
        self.assertIn("Always follow the selected spec file exactly.", content)
        self.assertIn("Keep the shared contract consistent across modes", content)
        self.assertIn("metadata header", content)
        self.assertIn("deterministic NPU coverage", content)
        self.assertIn("repeated runs of the same harness produce identical inputs", content)
        self.assertNotIn("accept only `--operator-file` at runtime", content)
        self.assertNotIn("build_operator_api(operator_module)", content)
        self.assertNotIn("build_differential_test_cases(operator_api)", content)

    def test_bench_gen_skill_requires_header_metadata_and_no_runtime_api_flag(self) -> None:
        content = _read("skills/triton-npu-gen-bench/SKILL.md")
        self.assertIn("# bench-mode:", content)
        self.assertIn("# api-name:", content)
        self.assertIn("# api-kind:", content)
        self.assertIn("# kernels:", content)
        self.assertIn("build_operator_api(operator_module)", content)
        self.assertIn("build_standalone_bench_cases(operator_api)", content)
        self.assertIn("import-only", content)
        self.assertIn("repeated runs of the same harness produce identical inputs", content)
        self.assertNotIn("accept only `--operator-file` at runtime for standalone mode", content)
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

    def test_generation_skills_prefer_model_entrypoint_over_wrapper_chain(self) -> None:
        for relative_path in ("skills/triton-npu-gen-test/SKILL.md", "skills/triton-npu-gen-bench/SKILL.md"):
            content = _read(relative_path)
            with self.subTest(path=relative_path):
                self.assertIn("When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper", content)
                self.assertIn("prefer the module class as the public entrypoint", content)
                self.assertIn("rather than selecting the intermediate wrapper function", content)

    def test_generation_and_optimize_skills_do_not_reference_removed_run_skills(self) -> None:
        self.assertNotIn("skill `test-run`", _read("skills/triton-npu-gen-test/SKILL.md"))
        self.assertNotIn("`bench-run`", _read("skills/triton-npu-gen-bench/SKILL.md"))
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertNotIn("run-test --input", optimize)
        self.assertNotIn("run-bench --input", optimize)

    def test_convert_skill_requires_reusing_existing_tests_before_regeneration(self) -> None:
        convert_skill = _read("skills/triton-npu-convert-pytorch-operator/SKILL.md")

        self.assertIn("If a suitable test already exists in the operator workspace, reuse it", convert_skill)
        self.assertIn("This includes existing standalone and differential test cases", convert_skill)
        self.assertIn("Do not create a new test when an existing suitable test can be reused", convert_skill)
        self.assertIn("unless the user explicitly asks to regenerate", convert_skill)

    def test_generation_skills_include_explicit_run_command_examples(self) -> None:
        test_gen = _read("skills/triton-npu-gen-test/SKILL.md")
        self.assertIn("## Validation Commands", test_gen)
        self.assertIn("Use the `triton-npu-run-eval` skill to validate generated tests.", test_gen)
        self.assertIn("run `run-test-baseline`", test_gen)
        self.assertIn("run `run-test-optimize`", test_gen)
        self.assertIn("run `run-test-optimize` with `--baseline-operator-file <baseline_operator.py>`", test_gen)
        self.assertIn("keep `compare-result` for reruns", test_gen)

        bench_gen = _read("skills/triton-npu-gen-bench/SKILL.md")
        self.assertIn("## Validation Commands", bench_gen)
        self.assertIn("Use the triton-npu-run-eval skill to execute generated benchmark cases.", bench_gen)
        self.assertIn("python3 ../triton-npu-run-eval/scripts/run-command.py run-bench --bench-file", bench_gen)

    def test_run_eval_skill_routes_to_focused_command_docs(self) -> None:
        skill = _read("skills/triton-npu-run-eval/SKILL.md")
        run_test = _read("skills/triton-npu-run-eval/references/run-test.md")
        run_bench = _read("skills/triton-npu-run-eval/references/run-bench.md")
        profile_bench = _read("skills/triton-npu-run-eval/references/profile-bench.md")
        compare_result = _read("skills/triton-npu-run-eval/references/compare-result.md")
        compare_perf = _read("skills/triton-npu-run-eval/references/compare-perf.md")

        self.assertIn("# Run-Eval Router", skill)
        self.assertIn("references/run-test.md", skill)
        self.assertIn("references/run-bench.md", skill)
        self.assertIn("references/profile-bench.md", skill)
        self.assertIn("references/compare-result.md", skill)
        self.assertIn("references/compare-perf.md", skill)
        self.assertIn("do not read unrelated command guides", skill)
        self.assertIn("do not reread Python files under `./scripts/`", skill)
        self.assertNotIn("## Run Test", skill)
        self.assertNotIn("## Run Bench", skill)
        self.assertNotIn("## Profile Bench", skill)
        self.assertNotIn("## Compare Differential Results", skill)
        self.assertNotIn("## Compare Performance Results", skill)
        self.assertFalse((REPO_ROOT / "skills" / "triton-npu-run-eval" / "run-test.md").exists())
        self.assertFalse((REPO_ROOT / "skills" / "triton-npu-run-eval" / "run-bench.md").exists())
        self.assertFalse((REPO_ROOT / "skills" / "triton-npu-run-eval" / "profile-bench.md").exists())
        self.assertFalse((REPO_ROOT / "skills" / "triton-npu-run-eval" / "compare-result.md").exists())
        self.assertFalse((REPO_ROOT / "skills" / "triton-npu-run-eval" / "compare-perf.md").exists())

        self.assertIn("run-test-baseline", run_test)
        self.assertIn("run-test-optimize", run_test)
        self.assertIn("Always pass both `--test-file` and `--operator-file`.", run_test)
        self.assertNotIn("--oracle-result", run_test)
        self.assertIn("--baseline-operator-file", run_test)
        self.assertIn("run-test-optimize` requires `--baseline-operator-file`", run_test)
        self.assertIn("--remote user@host:2222", run_test)

        self.assertIn("Always pass both `--bench-file` and `--operator-file`.", run_bench)
        self.assertIn("build_operator_api(operator_module)", run_bench)
        self.assertIn("build_standalone_bench_cases(operator_api)", run_bench)
        self.assertIn("--bench-mode msprof", run_bench)

        self.assertIn("--case-id <id>", profile_bench)
        self.assertIn("do not pass kernel filter arguments", profile_bench)
        self.assertIn("--keep-remote-workdir", profile_bench)

        self.assertIn("rerun or inspect the comparison separately from `run-test-optimize`", compare_result)
        self.assertIn("Prefer `run-test-optimize --baseline-operator-file ...`", compare_result)
        self.assertIn("--compare-level balanced", compare_result)

        self.assertIn("Avg improvement", compare_perf)
        self.assertIn("Geomean speedup", compare_perf)
        self.assertIn("authority for claimed benchmark deltas and speedups", compare_perf)

    def test_run_eval_skill_keeps_legacy_script_path(self) -> None:
        skill = _read("skills/triton-npu-run-eval/SKILL.md")

        self.assertIn("Use the bundled helper script in this skill", skill)
        self.assertIn("run-test-baseline", skill)
        self.assertIn("run-test-optimize", skill)
        self.assertIn("run-bench", skill)
        self.assertIn("profile-bench", skill)
        self.assertIn("call `python3 ./scripts/run-command.py <subcommand> ...` directly", skill)
        self.assertNotIn("use the corresponding MCP tool", skill)

    def test_run_eval_mcp_skill_is_tool_first_and_omits_compare_result(self) -> None:
        skill = _read("skills/triton-npu-run-eval-mcp/SKILL.md")

        self.assertIn("use the corresponding MCP tool", skill)
        self.assertIn("profile-report", skill)
        self.assertIn("compare-perf", skill)
        self.assertNotIn("compare-result", skill)
        self.assertNotIn("python3 ./scripts/run-command.py", skill)
        self.assertFalse(
            (
                REPO_ROOT
                / "skills"
                / "triton-npu-run-eval-mcp"
                / "references"
                / "compare-result.md"
            ).exists()
        )

    def test_eval_gen_skill_documents_direct_operator_repair_and_remote_validation(self) -> None:
        eval_gen = _read("skills/triton-npu-gen-eval-suite/SKILL.md")
        self.assertIn("repair the original operator file", eval_gen)
        self.assertIn("triton-npu-gen-test", eval_gen)
        self.assertIn("triton-npu-gen-bench", eval_gen)
        self.assertIn("triton-npu-run-eval", eval_gen)
        self.assertIn("Use the `triton-npu-run-eval` skill for correctness validation", eval_gen)
        self.assertIn("run-test-baseline", eval_gen)
        self.assertIn("Use the `triton-npu-run-eval` skill for benchmark validation", eval_gen)
        self.assertIn("carry the same remote flags", eval_gen)
        self.assertIn("Do not", eval_gen)
        self.assertIn("opt-round", eval_gen)

    def test_convert_skill_and_readme_document_convert_test_mode_support(self) -> None:
        convert_skill = _read("skills/triton-npu-convert-pytorch-operator/SKILL.md")
        readme = _read("README.md")

        self.assertTrue(
            (REPO_ROOT / "skills" / "triton-npu-convert-pytorch-operator" / "SKILL.md").exists()
        )
        self.assertIn("trailing input-helper block", convert_skill)
        self.assertIn("Do not execute the original input operator file", convert_skill)
        self.assertIn("correctness oracle", convert_skill)
        self.assertIn("standalone or differential test mode", convert_skill)
        self.assertIn("standalone or differential test file", convert_skill)
        self.assertIn("triton_<origin-name>.py", convert_skill)
        self.assertIn("## Converted Example", convert_skill)
        self.assertIn("@triton.jit", convert_skill)
        self.assertIn("def triton_add", convert_skill)
        self.assertIn("class ModelNew", convert_skill)
        self.assertIn("def get_inputs()", convert_skill)
        self.assertIn("def get_init_inputs()", convert_skill)
        self.assertNotIn("construct `ModelNew(*get_init_inputs())`", convert_skill)
        self.assertNotIn("call `model(*get_inputs())`", convert_skill)
        self.assertIn("Do not introduce unnecessary code.", convert_skill)
        self.assertIn("real Triton Ascend NPU kernel path", convert_skill)
        self.assertIn("PyTorch-facing wrapper or `torch.nn.Module` public API may remain", convert_skill)
        self.assertIn("A pure PyTorch rewrite does not satisfy this convert task", convert_skill)
        self.assertIn("Target Ascend NPU only", convert_skill)
        self.assertIn("Do not add CUDA-only, CPU-only, MPS, or generic multi-backend dispatch branches", convert_skill)
        self.assertNotIn("triton-npu-prepare-optimize-baseline", convert_skill)
        self.assertNotIn("reusable baseline", convert_skill.lower())
        self.assertNotIn("benchmark", convert_skill.lower())
        self.assertIn("`convert`", readme)
        self.assertIn("`convert-batch`", readme)
        self.assertNotIn("`gen-convert`", readme)
        self.assertIn("Triton NPU-backed PyTorch operator", readme)
        self.assertIn("standalone or differential correctness validation", readme)
        self.assertIn("`--test-mode standalone|differential`", readme)
        self.assertNotIn("preparing `baseline/`", readme)

    def test_optimize_baseline_preparation_uses_dedicated_skill(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        baseline = _read("skills/triton-npu-prepare-optimize-baseline/SKILL.md")
        baseline_submit_path = (
            REPO_ROOT / "skills" / "triton-npu-optimize-submit-baseline" / "SKILL.md"
        )
        round_submit_path = REPO_ROOT / "skills" / "triton-npu-optimize-submit-round" / "SKILL.md"
        start_round_path = REPO_ROOT / "skills" / "triton-npu-optimize-start-round" / "SKILL.md"
        self.assertTrue(baseline_submit_path.exists())
        self.assertTrue(round_submit_path.exists())
        self.assertTrue(start_round_path.exists())
        baseline_submit = baseline_submit_path.read_text(encoding="utf-8")
        round_submit = round_submit_path.read_text(encoding="utf-8")
        start_round = start_round_path.read_text(encoding="utf-8")
        readme = _read("README.md")

        self.assertTrue(
            (REPO_ROOT / "skills" / "triton-npu-prepare-optimize-baseline" / "SKILL.md").exists()
        )
        self.assertIn("triton-npu-prepare-optimize-baseline", optimize)
        self.assertIn("triton-npu-gen-test", baseline)
        self.assertIn("triton-npu-gen-bench", baseline)
        self.assertIn("triton-npu-run-eval", baseline)
        self.assertIn("triton-npu-optimize-submit-baseline", baseline)
        self.assertIn("triton-npu-optimize-submit-round", optimize)
        self.assertIn("triton-npu-optimize-start-round", optimize)
        self.assertNotIn("../triton-npu-run-eval/scripts/run-command.py", optimize)
        self.assertIn("Do not use this skill to generate missing harnesses", baseline_submit)
        self.assertIn("Do not start the next optimize round until this submission passes", round_submit)
        self.assertIn("Only one optimize round may be active at a time", start_round)
        self.assertIn("triton-npu-prepare-optimize-baseline", readme)
        self.assertIn("triton-npu-profile-operator", optimize)
        self.assertIn("triton-npu-analyze-round-performance", optimize)
        self.assertIn("triton-npu-optimize-knowledge", optimize)
        self.assertIn("torch-npu-optimize-knowledge", optimize)
        self.assertIn("classic-matmul.md", optimize)
        self.assertIn("`opt-round-N/perf-analysis.md`", optimize)

    def test_optimize_knowledge_skill_owns_generic_pattern_references(self) -> None:
        knowledge = _read("skills/triton-npu-optimize-knowledge/SKILL.md")
        index = _read("skills/triton-npu-optimize-knowledge/references/pattern_index.md")
        reference = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/classic-matmul.md"
        )

        self.assertIn("reference-only", knowledge)
        self.assertIn("does not define optimize workflow", knowledge)
        self.assertIn("pattern_index.md", knowledge)
        self.assertIn("classic-matmul", index)
        self.assertNotIn("argsort-avoid-aicpu-fallback", index)
        self.assertIn("matmul-like", reference)

    def test_optimize_knowledge_skill_owns_generic_symptom_references(self) -> None:
        knowledge = _read("skills/triton-npu-optimize-knowledge/SKILL.md")
        symptom_index = _read(
            "skills/triton-npu-optimize-knowledge/references/symptom_index.md"
        )
        symptom = _read(
            "skills/triton-npu-optimize-knowledge/references/symptoms/weak-pipeline-overlap.md"
        )

        self.assertIn("symptom_index.md", knowledge)
        self.assertIn("weak-pipeline-overlap", symptom_index)
        self.assertNotIn("unsupported-dtype-fallback", symptom_index)
        self.assertIn("## Evidence To Confirm", symptom)
        self.assertIn("## Candidate Pattern Directions", symptom)

    def test_torch_npu_optimize_knowledge_skill_owns_operator_level_pattern_references(
        self,
    ) -> None:
        knowledge = _read("skills/torch-npu-optimize-knowledge/SKILL.md")
        pattern_index = _read("skills/torch-npu-optimize-knowledge/references/pattern_index.md")
        pattern = _read(
            "skills/torch-npu-optimize-knowledge/references/patterns/argsort-avoid-aicpu-fallback.md"
        )

        self.assertIn("reference-only", knowledge)
        self.assertIn("Torch NPU", knowledge)
        self.assertIn("pattern references", knowledge)
        self.assertNotIn("symptom_index.md", knowledge)
        self.assertIn("argsort-avoid-aicpu-fallback", pattern_index)
        self.assertIn("torch.argsort()", pattern)
        self.assertFalse(
            (
                REPO_ROOT
                / "skills"
                / "torch-npu-optimize-knowledge"
                / "references"
                / "symptom_index.md"
            ).exists()
        )
        self.assertFalse(
            (
                REPO_ROOT
                / "skills"
                / "torch-npu-optimize-knowledge"
                / "references"
                / "symptoms"
                / "unsupported-dtype-fallback.md"
            ).exists()
        )

    def test_optimize_pattern_library_includes_classic_tiled_matmul(self) -> None:
        index = _read("skills/triton-npu-optimize-knowledge/references/pattern_index.md")
        reference = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/classic-matmul.md"
        )
        self.assertIn("classic-matmul", index)
        self.assertIn("manual matmul or K-reduction", index)
        self.assertIn("dtype-specialized or shape-specialized paths", index)
        self.assertIn("matmul-like", reference)
        self.assertIn("BLOCK_M", reference)
        self.assertIn("BLOCK_N", reference)
        self.assertIn("BLOCK_K", reference)
        self.assertIn("offs_m", reference)
        self.assertIn("offs_n", reference)
        self.assertIn("offs_k", reference)
        self.assertIn("acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)", reference)
        self.assertIn("a = tl.load(a_ptrs, mask=a_mask, other=0.0)", reference)
        self.assertIn("b = tl.load(b_ptrs, mask=b_mask, other=0.0)", reference)
        self.assertIn("do not lower them to `fp16` by default", reference)
        self.assertIn("sufficiently large `M`: tiled matmul path", reference)
        self.assertIn("small shapes: baseline-style reduction path", reference)

    def test_optimize_pattern_library_fuses_latency_optimizer_guidance(self) -> None:
        index = _read("skills/triton-npu-optimize-knowledge/references/pattern_index.md")
        scalar = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/scalar-latency-traps.md"
        )
        layout = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/block-pointer-dimensionality.md"
        )
        grid = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/grid-flatten-and-ub-buffering.md"
        )
        attention = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/attention-cv-pipeline.md"
        )

        self.assertIn("scalar-latency-traps", index)
        self.assertIn("merge-adjacent-stores", index)
        self.assertIn("grid-flatten-and-ub-buffering", index)
        self.assertIn("attention-cv-pipeline", index)
        self.assertIn("modulo addressing", index)
        self.assertIn("grid-to-physical-core mapping", index)

        self.assertIn("tl.constexpr", scalar)
        self.assertIn("Loop pointer recurrences", scalar)
        self.assertIn("Modulo removal", scalar)
        self.assertIn("Cumsum axis splitting", scalar)
        self.assertIn("flattened one-dimensional offsets", layout)
        self.assertIn("tl.make_block_ptr", layout)
        self.assertIn("boundary_check", layout)
        self.assertIn("physical cores", grid)
        self.assertIn("UB aggregate writes", grid)
        self.assertIn("UB bulk reads", grid)
        self.assertIn("Precompute repeated masks", attention)
        self.assertIn("scale and mask", attention)
        self.assertIn("A5", attention)

    def test_grid_flatten_pattern_documents_runtime_query_and_core_count_fallbacks(
        self,
    ) -> None:
        grid = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/grid-flatten-and-ub-buffering.md"
        )

        self.assertIn("torch.npu.get_device_properties", grid)
        self.assertIn("cube cores: `24`", grid)
        self.assertIn("vector cores: `48`", grid)
        self.assertIn("`cube`-like operators", grid)
        self.assertIn("`vector`-like operators", grid)
        self.assertIn("`mix` operators", grid)

    def test_optimize_pattern_cards_use_required_sections_and_generated_index(self) -> None:
        patterns_dir = (
            REPO_ROOT
            / "skills"
            / "triton-npu-optimize-knowledge"
            / "references"
            / "patterns"
        )
        for path in sorted(patterns_dir.glob("*.md")):
            if path.name == "index.md":
                continue
            content = path.read_text(encoding="utf-8")
            normalized = content
            if content.startswith("---\n"):
                _, _, normalized = content.partition("\n---\n")
                normalized = normalized.lstrip()
            with self.subTest(path=path.name):
                self.assertTrue(normalized.startswith("# "))
                self.assertIn("## Summary", content)
                self.assertIn("## Use When", content)

        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertIn("triton-npu-optimize-knowledge", optimize)
        self.assertIn(
            "../triton-npu-optimize-knowledge/references/pattern_index.md",
            optimize,
        )
        self.assertNotIn("extract_code_facts.py", optimize)

    def test_optimize_pattern_cards_promote_existing_information_into_structured_sections(
        self,
    ) -> None:
        program_rows = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/program-multiple-rows.md"
        )
        software_pipeline = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/software-pipeline.md"
        )
        tiling = _read("skills/triton-npu-optimize-knowledge/references/patterns/tiling.md")
        vec_cmp = _read("skills/triton-npu-optimize-knowledge/references/patterns/vec-cmp.md")
        discrete_memory = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/discrete_memory_access.md"
        )
        reorder_load = _read(
            "skills/triton-npu-optimize-knowledge/references/patterns/reorder-load.md"
        )

        self.assertIn("## Signals", program_rows)
        self.assertIn("## Avoid When", program_rows)
        self.assertIn("## What To Verify After Applying", program_rows)
        self.assertIn("## Related Patterns", program_rows)
        self.assertNotIn("## Symptoms (code + profiler)", program_rows)
        self.assertNotIn("## What not to do (common pitfalls)", program_rows)
        self.assertNotIn("## Verification checklist", program_rows)
        self.assertNotIn("## Relation to other patterns", program_rows)

        for content in (software_pipeline, tiling, vec_cmp, discrete_memory, reorder_load):
            with self.subTest(card_preview=content.splitlines()[0]):
                self.assertIn("## Signals", content)
                self.assertIn("## What To Verify After Applying", content)

        self.assertIn("## Related Patterns", software_pipeline)
        self.assertIn("## Related Patterns", tiling)
        self.assertIn("propagate_nan=tl.PropagateNan.ALL", vec_cmp)
        self.assertIn("NaN-input behavior", vec_cmp)

    def test_optimize_artifacts_reference_documents_state_declared_paths(self) -> None:
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")
        self.assertIn("<!-- BEGIN GENERATED BASELINE STATE CONTRACT -->", artifacts)
        self.assertIn("<!-- END GENERATED ROUND STATE CONTRACT -->", artifacts)
        self.assertIn("Treat these state fields as the authoritative artifact references for baseline validation:", artifacts)
        self.assertIn("`baseline_operator`", artifacts)
        self.assertIn("`perf_artifact`", artifacts)
        self.assertIn("Treat these round-state fields as the authoritative artifact references for round validation:", artifacts)
        self.assertIn("`summary_path`", artifacts)
        self.assertIn("`perf_analysis_path` when present", artifacts)
        self.assertIn("`profile_dir` when present", artifacts)
        self.assertIn("`ir_dir` when present", artifacts)
        self.assertIn("relative to the directory that contains `baseline/state.json`", artifacts)
        self.assertIn("relative to the directory that contains `round-state.json`", artifacts)
        self.assertNotIn(
            "The checker first resolves declared paths there. If a declared path is missing, it retries the same value relative to the operator workspace root",
            artifacts,
        )
        self.assertNotIn("analysis_comparison_sources", artifacts)
        self.assertNotIn("validated_candidate", artifacts)

    def test_submit_optimize_skills_keep_path_rules_out_of_user_facing_skill_text(self) -> None:
        baseline_submit = _read("skills/triton-npu-optimize-submit-baseline/SKILL.md")
        round_submit = _read("skills/triton-npu-optimize-submit-round/SKILL.md")

        self.assertNotIn("## Baseline-State Path Convention", baseline_submit)
        self.assertNotIn(
            "If a declared path is missing there, it retries the same value relative to the operator workspace root",
            baseline_submit,
        )
        self.assertNotIn(
            "If a declared path is missing there, it retries the same value relative to the operator workspace root",
            round_submit,
        )

    def test_optimize_artifacts_reference_matches_generator_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative_path in (
                "skills/triton-npu-optimize-submit-baseline/references/contract.json",
                "skills/triton-npu-optimize-submit-round/references/contract.json",
                "skills/triton-npu-optimize/references/artifacts.md",
                "skills/triton-npu-optimize/script/update-artifacts.py",
            ):
                source = REPO_ROOT / relative_path
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

            script_path = root / "skills" / "triton-npu-optimize" / "script" / "update-artifacts.py"
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                check=False,
                cwd=root,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            generated = (
                root / "skills" / "triton-npu-optimize" / "references" / "artifacts.md"
            ).read_text(encoding="utf-8")
            self.assertEqual(
                generated,
                _read("skills/triton-npu-optimize/references/artifacts.md"),
            )

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
        self.assertIn("owner of `opt-round-N/perf-analysis.md`", content)
        self.assertIn("`profile-only diagnosis`", content)
        self.assertIn("`profile-plus-IR diagnosis`", content)
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

    def test_round_performance_skill_points_to_knowledge_symptom_routing_references(
        self,
    ) -> None:
        skill = _read("skills/triton-npu-analyze-round-performance/SKILL.md")
        symptom_index = _read(
            "skills/triton-npu-optimize-knowledge/references/symptom_index.md"
        )
        self.assertIn("symptom cards", skill)
        self.assertIn("triton-npu-optimize-knowledge", skill)
        self.assertIn("torch-npu-optimize-knowledge", skill)
        self.assertIn(
            "../triton-npu-optimize-knowledge/references/symptom_index.md",
            skill,
        )
        self.assertIn(
            "../torch-npu-optimize-knowledge/references/pattern_index.md",
            skill,
        )
        self.assertIn("weak-pipeline-overlap", symptom_index)
        self.assertIn("high-transfer-pressure", symptom_index)

    def test_agents_declares_knowledge_skill_as_generic_pattern_and_symptom_source(
        self,
    ) -> None:
        agents = _read("AGENTS.md")
        self.assertIn("create-optimize-pattern", agents)
        skill = _read(".codex/skills/create-optimize-pattern/SKILL.md")
        self.assertIn(
            "skills/triton-npu-optimize-knowledge/references/patterns/",
            skill,
        )
        self.assertIn(
            "skills/torch-npu-optimize-knowledge/references/patterns/",
            skill,
        )
        self.assertIn(
            "skills/triton-npu-cann-ext-api-patterns/references/patterns/",
            skill,
        )
        self.assertIn(
            "skills/triton-npu-optimize-knowledge/references/symptoms/",
            skill,
        )
        self.assertIn("## Evidence To Confirm", skill)
        self.assertIn("## Candidate Pattern Directions", skill)

    def test_create_optimize_pattern_skill_declares_priority_rule(self) -> None:
        skill = _read(".codex/skills/create-optimize-pattern/SKILL.md")
        self.assertIn("priority: high|normal", skill)
        self.assertIn("default to `normal`", skill)
        self.assertIn("## High Priority Patterns", skill)

    def test_skill_points_to_shared_optimize_knowledge_index_update_script(self) -> None:
        skill = _read(".codex/skills/create-optimize-pattern/SKILL.md")
        script = _read("scripts/update-optimize-knowledge-indices.sh")

        self.assertIn("bash scripts/update-optimize-knowledge-indices.sh", skill)
        self.assertIn("build_pattern_index.py", script)
        self.assertIn("build_symptom_index.py", script)
        self.assertIn(
            "skills/triton-npu-optimize-knowledge/references/pattern_index.md",
            script,
        )
        self.assertIn(
            "skills/torch-npu-optimize-knowledge/references/pattern_index.md",
            script,
        )
        self.assertIn(
            "skills/triton-npu-cann-ext-api-patterns/references/patterns/index.md",
            script,
        )
        self.assertIn(
            "skills/triton-npu-optimize-knowledge/references/symptom_index.md",
            script,
        )

    def test_pattern_and_symptom_authoring_notes_point_to_knowledge_skill(self) -> None:
        pattern_note = _read("docs/notes/2026-04-29-optimize-pattern-card-authoring.md")
        symptom_note = _read("docs/notes/2026-04-30-optimize-symptom-card-authoring.md")

        self.assertIn(
            "skills/triton-npu-optimize-knowledge/references/patterns/",
            pattern_note,
        )
        self.assertIn(
            "skills/triton-npu-optimize-knowledge/scripts/build_pattern_index.py",
            pattern_note,
        )
        self.assertIn(
            "skills/triton-npu-optimize-knowledge/references/symptoms/",
            symptom_note,
        )
        self.assertIn("build_symptom_index.py", symptom_note)

    def test_pattern_authoring_note_describes_priority_contract(self) -> None:
        pattern_note = _read("docs/notes/2026-04-29-optimize-pattern-card-authoring.md")
        self.assertIn("priority: high|normal", pattern_note)
        self.assertIn("default to `normal`", pattern_note)
        self.assertIn("## High Priority Patterns", pattern_note)

    def test_compiler_source_analysis_skill_focuses_on_performance_navigation_and_next_action(
        self,
    ) -> None:
        content = _read("skills/triton-npu-analyze-compiler-source/SKILL.md")

        self.assertIn("Analyze Compiler Source For Performance", content)
        self.assertIn("Round Performance Question", content)
        self.assertIn("Round Evidence Used", content)
        self.assertIn("Recommended Next Operator Change", content)
        self.assertIn("references/navigation-map.md", content)
        self.assertIn("references/perf-question-playbook.md", content)
        self.assertIn("Inspect `<compiler-source-dir>/docs/` first", content)
        self.assertIn("`<compiler-source-dir>/bishengir/lib/` for implementation evidence", content)
        self.assertIn("`<compiler-source-dir>/bishengir/include/` only when declarations", content)
        self.assertIn(
            "`<compiler-source-dir>/bishengir/test/` only when a minimal example is genuinely necessary",
            content,
        )
        self.assertIn("Treat the compiler source checkout as read-only", content)
        self.assertIn("Do not run `git clone`, `git fetch`, or `git pull`", content)
        self.assertIn("CLI-provided compiler source path and commit", content)
        self.assertNotIn("compiler error", content.lower())

    def test_compiler_source_navigation_references_exist_and_capture_expected_sections(self) -> None:
        navigation = _read(
            "skills/triton-npu-analyze-compiler-source/references/navigation-map.md"
        )
        playbook = _read(
            "skills/triton-npu-analyze-compiler-source/references/perf-question-playbook.md"
        )

        self.assertIn("# Compiler Source Navigation Map", navigation)
        self.assertIn("## Default Reading Order", navigation)
        self.assertIn("round evidence -> <compiler-source-dir>/docs", navigation)
        self.assertIn("## Symptom To Subtree", navigation)
        self.assertIn("## Search Recipes", navigation)
        self.assertIn("## Anti-Patterns", navigation)

        self.assertIn("# Performance Question Playbook", playbook)
        self.assertIn("## Suspicious Stage Transition", playbook)
        self.assertIn("## Vectorization Loss", playbook)
        self.assertIn("## Copy Or Sync Growth", playbook)
        self.assertIn("## Turning Source Findings Into Operator Actions", playbook)

    def test_optimize_skills_document_compiler_source_escalation(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        round_analysis = _read("skills/triton-npu-analyze-round-performance/SKILL.md")

        self.assertIn("compiler-source escalation", optimize)
        self.assertIn("triton-npu-analyze-compiler-source", optimize)
        self.assertIn("compiler source analysis is enabled", optimize)
        self.assertIn("performance-focused explanation", optimize)
        self.assertIn("next operator change", optimize)
        self.assertIn("after profiler and IR evidence", optimize)
        self.assertIn("opt-round-N/compiler-analysis.md", optimize)
        self.assertIn("compiler source analysis is enabled", round_analysis)
        self.assertIn("performance-related compiler-side question", round_analysis)
        self.assertIn("next operator change", round_analysis)
        self.assertFalse(
            (REPO_ROOT / "skills" / "triton-npu-optimize" / "references" / "workflow.md").exists()
        )

    def test_readme_documents_compiler_source_analysis_options(self) -> None:
        content = _read("README.md")

        self.assertIn("--enable-compiler-source-analysis", content)
        self.assertIn("~/.triton-agent/compiler-sources/AscendNPU-IR/", content)
        self.assertIn("read-only", content)
        self.assertIn("escalation", content)

    def test_readme_documents_cann_ext_api_option(self) -> None:
        content = _read("README.md")

        self.assertIn("--enable-cann-ext-api", content)
        self.assertIn("A5", content)

    def test_optimize_skill_family_contains_cann_ext_api_pattern_skill(self) -> None:
        skill = _read("skills/triton-npu-cann-ext-api-patterns/SKILL.md")
        index = _read("skills/triton-npu-cann-ext-api-patterns/references/patterns/index.md")
        pattern = _read("skills/triton-npu-cann-ext-api-patterns/references/patterns/sub_vec_id_1to2.md")

        self.assertIn("CANN Triton extension API", skill)
        self.assertIn("A5", skill)
        self.assertIn("references/patterns/index.md", skill)
        self.assertIn("sub-vec-id-1to2", index)
        self.assertIn("sub_vec_id_1to2.md", index)
        self.assertIn("sub_vec_id", pattern)
        self.assertIn("Quick Start", pattern)

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

    def test_optimize_docs_keep_opt_note_round_only_and_put_initial_hypothesis_in_attempts(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        opt_note = _read("skills/triton-npu-optimize/references/opt-note-format.md")
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")

        self.assertIn("completed round entries and one final `## Overall Summary`", optimize)
        self.assertIn(
            "For round 1, record the initial round hypothesis in `opt-round-1/attempts.md`",
            optimize,
        )
        self.assertIn("top-level round ledger plus final `## Overall Summary`", optimize)
        self.assertIn(
            "completed round records and final outcome summary",
            opt_note,
        )
        self.assertIn(
            "Do not put session-start diagnosis, tentative bottleneck narrative, or other pre-round analysis above the round history",
            opt_note,
        )
        self.assertIn(
            "Do not write session-start diagnosis or tentative bottleneck narrative in `opt-note.md`",
            artifacts,
        )
        self.assertNotIn("Record a short diagnosis before the first code-changing round", optimize)
        self.assertFalse(
            (REPO_ROOT / "skills" / "triton-npu-optimize" / "references" / "workflow.md").exists()
        )

    def test_optimize_docs_make_layered_analysis_default_and_remove_require_analysis_flag(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")
        readme = _read("README.md")

        self.assertIn("## Core Loop", optimize)
        self.assertIn("## Stage 2: Layered Analysis", optimize)
        self.assertIn("### pattern triage", optimize)
        self.assertIn("### profiling diagnosis", optimize)
        self.assertIn("### IR attribution", optimize)
        self.assertIn("### compiler-source escalation", optimize)
        self.assertIn(
            "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough",
            optimize,
        )
        self.assertIn("Primary analysis level", optimize)
        self.assertIn("Supporting evidence", optimize)
        self.assertIn(
            "`triton-npu-analyze-round-performance` may still own `opt-round-N/perf-analysis.md`",
            optimize,
        )
        self.assertIn("Escalation: <from> -> <to>", optimize)
        self.assertIn("Escalation reason:", optimize)
        self.assertIn("the current analysis level", artifacts)
        self.assertIn("why the round stayed at that level or why it escalated deeper", artifacts)
        self.assertIn("Primary analysis level", artifacts)
        self.assertIn("Supporting evidence", artifacts)
        self.assertIn(
            "pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation",
            readme,
        )
        self.assertNotIn("--require-analysis", readme)
        self.assertFalse(
            (REPO_ROOT / "skills" / "triton-npu-optimize" / "references" / "workflow.md").exists()
        )

    def test_optimize_docs_record_pattern_choice_in_attempts_and_summary(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")

        self.assertIn("selected pattern candidates and pivots", optimize)
        self.assertIn("final selected pattern direction", optimize)
        self.assertIn("candidate patterns considered at the start of the round", artifacts)
        self.assertIn("the selected pattern and why it looked plausible", artifacts)
        self.assertIn("pattern pivots", artifacts)
        self.assertIn("final selected pattern direction", artifacts)
        self.assertIn("pivoted away from an earlier pattern choice", artifacts)

    def test_optimize_skill_declares_layered_analysis_and_deduplicates_compare_perf_and_lessons(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")

        self.assertIn("Optimize analysis is layered.", optimize)
        self.assertIn(
            "Default escalation order: `pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation`.",
            optimize,
        )
        self.assertIn(
            "Start each round at the shallowest level that can justify the next move.",
            optimize,
        )
        self.assertLess(optimize.index("Optimize analysis is layered."), optimize.index("### pattern triage"))
        self.assertEqual(
            optimize.count("use the `triton-npu-run-eval` skill to run `compare-perf`"),
            1,
        )
        self.assertIn("## Learned Lessons", optimize)
        self.assertIn("Admission criteria:", optimize)
        self.assertIn("Put round-local narrative", optimize)

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
        self.assertIn("generation-only workflow such as `triton-npu-gen-test`", repair_guide)
        self.assertFalse((REPO_ROOT / "skills" / "triton-npu-log-repair").exists())

    def test_generation_skills_treat_repair_guide_as_diagnostic_only_reference(self) -> None:
        test_gen = _read("skills/triton-npu-gen-test/SKILL.md")
        bench_gen = _read("skills/triton-npu-gen-bench/SKILL.md")

        self.assertIn("consult the `triton-npu-repair-guide` skill as a diagnostic reference", test_gen)
        self.assertIn("Do not treat `triton-npu-repair-guide` as permission to edit the operator file here.", test_gen)
        self.assertIn("If the failure is clearly operator-side, stop and report it.", test_gen)

        self.assertIn("consult the `triton-npu-repair-guide` skill as a diagnostic reference", bench_gen)
        self.assertIn("Do not treat `triton-npu-repair-guide` as permission to edit the operator file here.", bench_gen)
        self.assertIn("If the failure is clearly operator-side, stop and report it.", bench_gen)

    def test_cross_skill_subcommands_name_owning_skills(self) -> None:
        optimize = _read("skills/triton-npu-optimize/SKILL.md")
        self.assertIn("use the `triton-npu-run-eval` skill to run `compare-perf`", optimize)
        self.assertIn("the `triton-npu-run-eval` skill's `compare-perf` flow", optimize)
        self.assertIn("effective_metric_source", optimize)
        self.assertIn("show both kernel and total-op comparison results", optimize)

    def test_compare_perf_reference_documents_all_mode_and_optimize_target_usage(self) -> None:
        compare_perf = _read("skills/triton-npu-run-eval/references/compare-perf.md")
        self.assertIn("--metric-source auto|kernel|total-op|all", compare_perf)
        self.assertIn("operator-target optimize rounds", compare_perf)
        self.assertIn("effective_metric_source", compare_perf)

    def test_optimize_artifacts_reference_documents_effective_metric_source(self) -> None:
        artifacts = _read("skills/triton-npu-optimize/references/artifacts.md")
        self.assertIn("effective_metric_source", artifacts)
        self.assertIn("kernel", artifacts)
        self.assertIn("total-op", artifacts)
        self.assertIn("mixed", artifacts)

    def test_profiler_skill_documents_standalone_case_id_contract(self) -> None:
        profiler = _read("skills/triton-npu-profile-operator/SKILL.md")
        self.assertIn("../triton-npu-run-eval/scripts/run-command.py profile-bench", profiler)
        self.assertIn("standalone", profiler)
        self.assertIn("msprof", profiler)
        self.assertIn("--case-id <id>", profiler)
        self.assertIn("profile one selected `--case-id <id>` case", profiler)
        self.assertIn("must not receive `--bench` or `--num-bench`", profiler)
        self.assertIn("first query `--num-bench`", profiler)
        self.assertNotIn("profile one selected `--bench <N>` case", profiler)
        self.assertNotIn("msprof python3 bench_<op>.py --operator-file <operator-file>", profiler)

    def test_test_generation_specs_distinguish_standalone_cli_and_differential_hooks(self) -> None:
        standalone = _read("skills/triton-npu-gen-test/references/test-standalone-spec.md")
        differential = _read("skills/triton-npu-gen-test/references/test-differential-spec.md")

        for content in (standalone, differential):
            with self.subTest(spec=content[:40]):
                self.assertIn("# api-name: <name>", content)
                self.assertIn("# api-kind: <triton-wrapper|torch-function|torch-module>", content)
                self.assertIn("# kernels: <name>", content)
                self.assertIn("# api-name: <resolved_entrypoint>", content)
                self.assertIn("# api-kind: <resolved_api_kind>", content)
                self.assertIn("# kernels: <resolved_kernel_names>", content)
                self.assertNotIn("| `--api-name <name>` | yes |", content)
                self.assertIn("triton-wrapper", content)
                self.assertIn("torch-function", content)
                self.assertIn("torch-module", content)

        self.assertIn("Parses `--operator-file`", standalone)
        self.assertIn('parser.add_argument("--operator-file", required=True)', standalone)

        self.assertIn("import-only", differential)
        self.assertIn("build_operator_api(operator_module)", differential)
        self.assertIn("build_differential_test_cases(operator_api)", differential)
        self.assertIn("The runner loads the operator module by file path", differential)
        self.assertIn("<operator>_result.pt", differential)
        self.assertNotIn("TEST_RESULT.pt", differential)
        self.assertNotIn("Parses `--operator-file`", differential)
        self.assertNotIn('parser.add_argument("--operator-file"', differential)
        self.assertNotIn("def main()", differential)
        self.assertNotIn("Running the file directly should execute the differential test", differential)

    def test_benchmark_generation_specs_use_hooked_standalone_contract(self) -> None:
        standalone = _read("skills/triton-npu-gen-bench/references/bench-standalone-spec.md")
        msprof = _read("skills/triton-npu-gen-bench/references/bench-msprof-spec.md")

        self.assertIn("# bench-mode: standalone", standalone)
        self.assertIn("# api-name: <resolved_entrypoint>", standalone)
        self.assertIn("# api-kind: <resolved_api_kind>", standalone)
        self.assertIn("# kernels: <resolved_kernel_names>", standalone)
        self.assertIn("repeated runs of the same harness produce identical inputs", standalone)
        self.assertNotIn("| `--api-name <name>` | yes |", standalone)
        self.assertIn("build_operator_api(operator_module)", standalone)
        self.assertIn("build_standalone_bench_cases(operator_api)", standalone)
        self.assertIn("import-only", standalone)
        self.assertNotIn('parser.add_argument("--operator-file"', standalone)
        self.assertNotIn("def run_bench(operator_api):", standalone)
        self.assertNotIn('print(f"latency-{case_id}: {latency}")', standalone)
        self.assertIn("must be **<= 20**", standalone)
        self.assertIn("prefer **8-20 representative cases**", standalone)
        self.assertIn("cover small, medium, and large representative shapes", standalone)
        self.assertIn("torch-module", standalone)
        self.assertIn("constructor arguments", standalone)

        self.assertIn("# bench-mode: msprof", msprof)
        self.assertIn("# api-name: <resolved_entrypoint>", msprof)
        self.assertIn("# api-kind: <resolved_api_kind>", msprof)
        self.assertIn("# kernels: <resolved_kernel_names>", msprof)
        self.assertIn("repeated runs of the same harness produce identical inputs", msprof)
        self.assertNotIn("--api-name <api-name>", msprof)
        self.assertIn("If `--bench N` is provided, then `--operator-file` is required.", msprof)
        self.assertIn("torch-function", msprof)
        self.assertIn("must be **<= 20**", msprof)
        self.assertIn("prefer **8-20 representative cases**", msprof)
        self.assertIn("cover small, medium, and large representative shapes", msprof)
        self.assertIn("**Warmup:** run the kernel **5 times**", msprof)
        self.assertIn("**Repeat:** after warmup, run the kernel **50 times**", msprof)
        self.assertIn("MSPROF_REPEAT_ITERS = 50", msprof)
        self.assertIn("for _ in range(MSPROF_REPEAT_ITERS):", msprof)

    def test_contracts_do_not_depend_on_workspace_placeholder_examples(self) -> None:
        test_spec = _read("skills/triton-npu-gen-test/references/test-standalone-spec.md")
        bench_spec = _read("skills/triton-npu-gen-bench/references/bench-standalone-spec.md")

        self.assertIn("# test-mode:", test_spec)
        self.assertIn("# api-name:", test_spec)
        self.assertIn("# api-kind:", test_spec)
        self.assertIn("# kernels:", test_spec)
        self.assertIn("repeated runs of the same harness produce identical inputs", test_spec)
        self.assertIn('parser.add_argument("--operator-file", required=True)', test_spec)
        self.assertNotIn('parser.add_argument("--api-name"', test_spec)

        self.assertIn("# bench-mode:", bench_spec)
        self.assertIn("# api-name:", bench_spec)
        self.assertIn("# api-kind:", bench_spec)
        self.assertIn("# kernels:", bench_spec)
        self.assertIn("repeated runs of the same harness produce identical inputs", bench_spec)
        self.assertIn("build_operator_api(operator_module)", bench_spec)
        self.assertIn("build_standalone_bench_cases(operator_api)", bench_spec)
        self.assertNotIn('parser.add_argument("--operator-file"', bench_spec)
        self.assertNotIn('parser.add_argument("--api-name"', bench_spec)
        self.assertNotIn('print(f"latency-', bench_spec)


if __name__ == "__main__":
    unittest.main()
