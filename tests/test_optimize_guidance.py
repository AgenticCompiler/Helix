import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import triton_agent.optimize.memory_file as optimize_memory_file
from triton_agent.optimize.session_artifacts import OptimizeSessionArtifactsManager


class OptimizeSessionArtifactsManagerTests(unittest.TestCase):
    def test_memory_file_manager_selects_agents_by_default(self) -> None:
        from triton_agent.optimize.memory_file import MemoryFileManager

        manager = MemoryFileManager()

        self.assertEqual(manager.guidance_filename("codex"), "AGENTS.md")

    def test_memory_file_manager_selects_claude_memory_file(self) -> None:
        from triton_agent.optimize.memory_file import MemoryFileManager

        manager = MemoryFileManager()

        self.assertEqual(manager.guidance_filename("claude"), "CLAUDE.md")

    def test_supervised_session_creates_and_cleans_runtime_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            manager = OptimizeSessionArtifactsManager()

            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
            )

            assert state.supervisor_report_path is not None
            assert state.supervisor_history_dir is not None
            self.assertTrue(state.supervisor_report_path.exists())
            self.assertTrue(state.supervisor_history_dir.exists())

            warnings = manager.cleanup_supervised_session(state)

            self.assertEqual(warnings, [])
            self.assertFalse((workdir / ".triton-agent").exists())

    def test_archive_manager_builds_run_paths(self) -> None:
        from triton_agent.optimize.archive import ArchiveManager

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            manager = ArchiveManager(run_id_factory=lambda: "20260423-123456-000000")

            state = manager.prepare(workdir, include_shared_guidance_snapshot=True)

            expected_run_dir = workdir / "triton-agent-logs" / "20260423-123456-000000"
            self.assertEqual(state.run_archive_dir, expected_run_dir)
            self.assertEqual(
                state.agent_session_path("baseline"),
                state.run_archive_dir / "agent-session-baseline.json",
            )
            self.assertEqual(
                state.trace_path("batch-1-5"),
                state.run_archive_dir / "trace-batch-1-5.jsonl",
            )
            self.assertEqual(
                state.trace_summary_path("supervisor"),
                state.run_archive_dir / "trace-supervisor.summary.json",
            )
            self.assertEqual(
                state.shared_guidance_snapshot_path,
                state.run_archive_dir / "shared-guidance.md",
            )

    def test_archive_manager_records_agent_session_compact_json(self) -> None:
        from triton_agent.optimize.archive import ArchiveManager

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            manager = ArchiveManager(run_id_factory=lambda: "20260423-123456-000000")
            state = manager.prepare(workdir)

            warning = manager.record_agent_session(
                state,
                label="batch-1-5",
                session_id="019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5",
                agent="codex",
            )

            self.assertIsNone(warning)
            payload = json.loads(state.agent_session_path("batch-1-5").read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"timestamp", "session_id", "agent"})

    def test_render_bullet_block_formats_markdown_list(self) -> None:
        rendered = optimize_memory_file._render_bullet_block(
            [
                "Read files cautiously.",
                "Follow the user's instructions strictly.",
            ]
        )

        self.assertEqual(
            rendered,
            "- Read files cautiously.\n"
            "- Follow the user's instructions strictly.\n",
        )

    def test_render_line_block_joins_lines_and_omits_empty_block(self) -> None:
        self.assertEqual(
            optimize_memory_file._render_line_block(
                [
                    "Compiler source analysis is enabled for this optimize run.",
                    "Treat the compiler source checkout as read-only.",
                ]
            ),
            "Compiler source analysis is enabled for this optimize run.\n"
            "Treat the compiler source checkout as read-only.\n",
        )
        self.assertEqual(optimize_memory_file._render_line_block([]), "")

    def test_shared_guidance_template_inlines_shared_rule_block(self) -> None:
        template = optimize_memory_file._SHARED_GUIDANCE_TEMPLATE

        self.assertIn("{guidance_filename}", template)
        self.assertIn("{analysis_block}", template)
        self.assertIn("{compiler_source_block}", template)
        self.assertIn(
            "- Read files cautiously. Do not read unrelated files speculatively or just in case.\n",
            template,
        )
        self.assertNotIn("{guidance_rules_block}", template)

    def test_prepare_checked_session_rejects_subagent_on_unsupported_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()

            with self.assertRaisesRegex(
                RuntimeError,
                "Optimize subagent staging only supports `codex`, `claude`, and `opencode`; got `pi`.",
            ):
                manager.prepare_checked_session(
                    workdir,
                    agent_name="pi",
                    enable_subagent=True,
                )

            self.assertFalse((workdir / "AGENTS.md").exists())
            self.assertFalse((workdir / ".pi").exists())

    def test_prepare_checked_session_stages_perf_diagnosis_subagent_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
                enable_subagent=True,
            )

            agents_path = workdir / "AGENTS.md"
            guidance_content = agents_path.read_text(encoding="utf-8")
            agent_path = workdir / ".codex" / "agents" / "triton-agent-perf-diagnosis-advisor.toml"

            self.assertTrue(agent_path.exists())
            self.assertIn(
                "A diagnosis subagent named `triton-agent-perf-diagnosis-advisor` is available in this workspace.",
                guidance_content,
            )
            self.assertIn(
                "Use it proactively when the bottleneck hypothesis is still unclear before deeper optimize edits.",
                guidance_content,
            )
            self.assertIn("must not perform optimization work", guidance_content)

            warnings = manager.cleanup_checked_session(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agent_path.exists())
            self.assertFalse((workdir / ".codex").exists())

    def test_prepare_checked_session_creates_round_loop_guidance_file_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
            )

            agents_path = workdir / "AGENTS.md"
            guidance_content = agents_path.read_text(encoding="utf-8")

            self.assertTrue(agents_path.exists())
            self.assertEqual(state.guidance_path, agents_path)
            self.assertIsNone(state.backup_path)
            self.assertTrue(state.created_guidance)
            self.assertIn("## Triton Agent Optimize Round Loop", guidance_content)
            self.assertIn("This workspace is under an optimize round loop.", guidance_content)
            self.assertIn(
                "Read files cautiously. Do not read unrelated files speculatively or just in case.",
                guidance_content,
            )
            self.assertIn(
                "Follow the user's instructions strictly.",
                guidance_content,
            )
            self.assertIn("Use the staged workspace skills as the workflow source of truth.", guidance_content)
            self.assertIn(
                "The CLI will inject previous round validation results directly into the next worker prompt when another round is needed.",
                guidance_content,
            )
            self.assertNotIn("Read the role brief", guidance_content)
            self.assertNotIn("Worker and supervisor roles", guidance_content)
            self.assertNotIn(".triton-agent/roles/", guidance_content)

            warnings = manager.cleanup_checked_session(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())
            self.assertFalse((workdir / ".triton-agent").exists())

    def test_prepare_checked_session_bootstraps_workflow_state_only_when_hooks_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            manager = OptimizeSessionArtifactsManager()

            state_without_hooks = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
                enable_agent_hooks=False,
                source_operator_path=operator,
            )
            self.assertIsNone(state_without_hooks.workflow_state_path)
            self.assertFalse((workdir / ".triton-agent").exists())
            warnings = manager.cleanup_checked_session(state_without_hooks)
            self.assertEqual(warnings, [])

            state_with_hooks = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
                enable_agent_hooks=True,
                source_operator_path=operator,
            )
            self.assertEqual(state_with_hooks.workflow_state_path, workdir / ".triton-agent" / "state.json")
            assert state_with_hooks.workflow_state_path is not None
            self.assertTrue(state_with_hooks.workflow_state_path.exists())

            warnings = manager.cleanup_checked_session(state_with_hooks)
            self.assertEqual(warnings, [])
            self.assertFalse((workdir / ".triton-agent").exists())

    def test_prepare_creates_shared_guidance_and_handoff_files_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
            )

            agents_path = workdir / "AGENTS.md"
            shared_content = agents_path.read_text(encoding="utf-8")
            self.assertTrue(agents_path.exists())
            self.assertEqual(state.guidance_path, agents_path)
            self.assertIsNone(state.backup_path)
            assert state.supervisor_report_path is not None
            assert state.supervisor_history_dir is not None
            self.assertTrue(state.supervisor_report_path.exists())
            self.assertTrue(state.supervisor_history_dir.exists())
            self.assertEqual(state.run_archive_dir.parent, workdir / "triton-agent-logs")
            self.assertEqual(
                state.agent_session_path("baseline"),
                state.run_archive_dir / "agent-session-baseline.json",
            )
            self.assertEqual(
                state.trace_path("supervisor"),
                state.run_archive_dir / "trace-supervisor.jsonl",
            )

            self.assertIn("## Triton Agent Optimize Orchestration", shared_content)
            self.assertIn("This workspace is under optimize orchestration.", shared_content)
            self.assertIn("Use the staged workspace skills as the workflow source of truth.", shared_content)
            self.assertIn("Invocation-specific behavior comes from the launch prompt.", shared_content)
            self.assertIn(
                "Use `supervisor-report.md` as the supervisor audit report file when supervised mode is active.",
                shared_content,
            )
            self.assertNotIn(".triton-agent/supervisor-report.md", shared_content)
            self.assertNotIn(".triton-agent/state.json", shared_content)
            self.assertIn("Treat `baseline/` as the canonical optimize baseline", shared_content)
            self.assertIn("Use `compare-perf` as the authoritative source", shared_content)
            self.assertNotIn("Improve the Triton operator", shared_content)
            self.assertNotIn("This invocation is an audit and handoff pass", shared_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])
            self.assertFalse(agents_path.exists())
            self.assertFalse((workdir / "supervisor-report.md").exists())
            self.assertFalse((workdir / ".triton-agent").exists())
            self.assertTrue(state.run_archive_dir.exists())
            self.assertTrue((state.run_archive_dir / "shared-guidance.md").exists())
            self.assertTrue((state.run_archive_dir / "supervisor-report.md").exists())
            self.assertTrue((state.run_archive_dir / "history").exists())

    def test_cleanup_supervised_session_writes_round_timings_archive_for_passed_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
                enable_agent_hooks=True,
                source_operator_path=operator,
            )

            assert state.workflow_state_path is not None
            state.workflow_state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": state.archive.run_id,
                        "phase": "round_active",
                        "source_operator": "kernel.py",
                        "current_round": 2,
                        "baseline": {"status": "passed", "submitted_at": "2026-06-22T12:34:56Z"},
                        "rounds": {
                            "1": {
                                "status": "passed",
                                "round_dir": "opt-round-1",
                                "started_at": "2026-06-22T12:40:00Z",
                                "ended_at": "2026-06-22T12:55:00Z",
                            },
                            "2": {
                                "status": "active",
                                "round_dir": "opt-round-2",
                                "started_at": "2026-06-22T13:10:00Z",
                                "ended_at": None,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            warnings = manager.cleanup_supervised_session(state)

            self.assertEqual(warnings, [])
            payload = json.loads((state.run_archive_dir / "round-timings.json").read_text(encoding="utf-8"))
            self.assertEqual([row["round"] for row in payload], [1])

    def test_prepare_checked_session_mentions_operator_target_when_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
                optimize_target="operator",
            )

            guidance_content = (workdir / "AGENTS.md").read_text(encoding="utf-8")
            self.assertIn("Target optimization scope: operator.", guidance_content)
            self.assertIn("Optimize end-to-end operator latency.", guidance_content)
            self.assertIn("both kernel and total-op `compare-perf` views visible", guidance_content)

            warnings = manager.cleanup_checked_session(state)
            self.assertEqual(warnings, [])

    def test_record_agent_session_writes_compact_json_per_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
            )

            manager.record_agent_session(
                state,
                label="batch-1-5",
                session_id="019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5",
                agent="codex",
            )

            payload = json.loads(
                state.agent_session_path("batch-1-5").read_text(encoding="utf-8")
            )
            self.assertEqual(set(payload), {"timestamp", "session_id", "agent"})
            self.assertEqual(payload["session_id"], "019da9c2-dfcb-7c71-a2f9-7a90bab2e0f5")
            self.assertEqual(payload["agent"], "codex")

            warnings = manager.cleanup_checked_session(state)
            self.assertEqual(warnings, [])
            self.assertTrue(state.agent_session_path("batch-1-5").exists())

    def test_record_agent_session_uses_unknown_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
            )

            manager.record_agent_session(
                state,
                label="supervisor",
                session_id=None,
                agent="codex",
            )

            payload = json.loads(state.agent_session_path("supervisor").read_text(encoding="utf-8"))
            self.assertEqual(payload["session_id"], "unknown")

    def test_prepare_uses_claude_file_and_restores_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            guidance_path = workdir / "CLAUDE.md"
            guidance_path.write_text("original content\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="claude",
            )

            shared_content = guidance_path.read_text(encoding="utf-8")
            self.assertIsNotNone(state.backup_path)
            self.assertTrue(state.backup_path is not None and state.backup_path.exists())
            self.assertEqual(state.guidance_path, guidance_path)
            self.assertIn("# CLAUDE.md", shared_content)
            self.assertIn("## Triton Agent Optimize Orchestration", shared_content)
            self.assertIn(
                "Read files cautiously. Do not read unrelated files speculatively or just in case.",
                shared_content,
            )
            self.assertIn(
                "Follow the user's instructions strictly.",
                shared_content,
            )
            self.assertIn("Invocation-specific behavior comes from the launch prompt.", shared_content)
            self.assertNotIn("Improve the Triton operator", shared_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])
            self.assertEqual(guidance_path.read_text(encoding="utf-8"), "original content\n")
            self.assertFalse(state.backup_path is not None and state.backup_path.exists())

    def test_prepare_shared_guidance_defaults_to_layered_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
            )

            shared_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn(
                "Choose the analysis level for each round before editing code.",
                shared_content,
            )
            self.assertIn(
                "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
                shared_content,
            )
            self.assertIn(
                "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.",
                shared_content,
            )
            self.assertIn(
                "Record the round's primary analysis level separately from its supporting evidence.",
                shared_content,
            )
            self.assertIn(
                "Use the staged `triton-npu-optimize-knowledge` skill for generic pattern and symptom references.",
                shared_content,
            )
            self.assertNotIn("torch-npu-optimize-knowledge", shared_content)
            self.assertIn(
                "When pattern triage is used, record candidate patterns, the selected pattern if one is chosen, and why that pattern looks plausible in `opt-round-N/attempts.md`.",
                shared_content,
            )
            self.assertIn(
                "When a named pattern guides the round, record the final selected pattern direction in `opt-round-N/summary.md`.",
                shared_content,
            )
            self.assertIn(
                "Read the staged `triton-npu-optimize-knowledge` skill's generated `references/pattern_index.md` before detailed pattern references.",
                shared_content,
            )
            self.assertIn(
                "Inspect the operator file directly when code structure is still unclear at pattern triage.",
                shared_content,
            )
            self.assertIn(
                "Use the staged `triton-npu-optimize-knowledge` skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists.",
                shared_content,
            )
            self.assertIn("Do not begin with blind tiling or launch-parameter search.", shared_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_shared_guidance_includes_generated_high_priority_pattern_reminders(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
                optimize_knowledge_skill_name="triton-npu-optimize-knowledge-v2",
            )

            shared_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn(
                "High-priority generic pattern reminders for this run:",
                shared_content,
            )
            self.assertIn(
                "`grid-flatten-and-ub-buffering`: Use this pattern when performance is limited by too many logical tasks, uneven per-core work, or tiny per-program transfers after a gather/scatter-style rewrite.",
                shared_content,
            )
            self.assertIn(
                "`autotune`: **Autotune** here means Triton’s `@triton.autotune` decorator: the runtime tries a **small, bounded** list of launch configurations (tile sizes, warp counts, pipeline stages, and other meta-parameters) and picks one that performs best on measured micro-benchmarks of the kernel.",
                shared_content,
            )

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_shared_guidance_mentions_torch_npu_knowledge_for_operator_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
                optimize_target="operator",
            )

            shared_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn(
                "Use the staged `torch-npu-optimize-knowledge` skill for Torch NPU and operator-level pattern references.",
                shared_content,
            )

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_checked_mentions_compiler_source_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            source_path = workdir / "AscendNPU-IR"
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
                compiler_source_path=source_path,
                compiler_source_commit="abc123",
            )

            guidance_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn("Compiler source analysis is enabled", guidance_content)
            self.assertIn(f"Compiler source path: {source_path.as_posix()}", guidance_content)
            self.assertIn("Compiler source commit: abc123.", guidance_content)
            self.assertIn("Treat the compiler source checkout as read-only.", guidance_content)
            self.assertIn("Do not run git clone, git fetch, git pull", guidance_content)
            self.assertNotIn("https://gitcode.com/Ascend/AscendNPU-IR.git", guidance_content)

            warnings = manager.cleanup_checked_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_checked_mentions_cann_ext_api_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_checked_session(
                workdir,
                agent_name="codex",
                enable_cann_ext_api=True,
            )

            guidance_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn("CANN Triton extension API pattern access is enabled", guidance_content)
            self.assertIn("triton-npu-cann-ext-api-patterns", guidance_content)
            self.assertIn("high-value optimization direction", guidance_content)
            self.assertIn("Give serious attention", guidance_content)

            warnings = manager.cleanup_checked_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_supervised_mentions_compiler_source_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            source_path = workdir / "AscendNPU-IR"

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
                compiler_source_path=source_path,
                compiler_source_commit="abc123",
            )

            guidance_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn("Compiler source analysis is enabled", guidance_content)
            self.assertIn(f"Compiler source path: {source_path.as_posix()}", guidance_content)
            self.assertIn("Compiler source commit: abc123.", guidance_content)
            self.assertIn("Treat the compiler source checkout as read-only.", guidance_content)
            self.assertIn("then IR evidence, then compiler source", guidance_content)
            self.assertNotIn("https://gitcode.com/Ascend/AscendNPU-IR.git", guidance_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_supervised_mentions_cann_ext_api_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
                enable_cann_ext_api=True,
            )

            guidance_content = state.guidance_path.read_text(encoding="utf-8")
            self.assertIn("CANN Triton extension API pattern access is enabled", guidance_content)
            self.assertIn("triton-npu-cann-ext-api-patterns", guidance_content)
            self.assertIn("high-value optimization direction", guidance_content)
            self.assertIn("Give serious attention", guidance_content)

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])

    def test_prepare_rejects_preexisting_nonempty_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")
            runtime_root = workdir / ".triton-agent"
            runtime_root.mkdir()
            (runtime_root / "leftover.txt").write_text("busy\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()

            with self.assertRaisesRegex(RuntimeError, "Existing \\.triton-agent/ directory contains data"):
                manager.prepare_supervised_session(
                    workdir,
                    agent_name="codex",
                )

    def test_describe_cleanup_lists_archive_and_runtime_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            operator = workdir / "kernel.py"
            operator.write_text("print('x')\n", encoding="utf-8")

            manager = OptimizeSessionArtifactsManager()
            state = manager.prepare_supervised_session(
                workdir,
                agent_name="codex",
            )

            messages = manager.describe_cleanup_supervised_session(state)

            self.assertTrue(any("archiving supervised optimize logs" in message for message in messages))
            self.assertTrue(any("supervisor-report.md" in message for message in messages))
            self.assertTrue(
                any("removing temporary optimize runtime directory tree" in message for message in messages)
            )

            warnings = manager.cleanup_supervised_session(state)
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
