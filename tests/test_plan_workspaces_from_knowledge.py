import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-pattern-validation-loop"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

from plan_workspaces_from_knowledge import main

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_KB = """\
## File Analyses

### src/kernels/fla/ops/common/chunk_o.py

#### Commit Timeline

##### 85171374766b [Perf] P0: Pre-transpose g in chunk_bwd_kernel_dv_local
- Classification: performance-related
- What changed: tuned `chunk_bwd_kernel_dv_local` grid layout.

##### 82dcaed82119 chunk_bwd_kernel_dqkwg离散访存
- Classification: performance-related
- What changed: converted `chunk_bwd_kernel_dqkwg` to contiguous access.
"""

SAMPLE_SOURCE = '''
import triton

@triton.jit
def chunk_bwd_kernel_dv_local(q):
    pass

@triton.jit
def chunk_bwd_kernel_dqkwg(q):
    pass

def chunk_bwd_dv_local(q, k, do):
    chunk_bwd_kernel_dv_local[(1,)](q=q)

def chunk_bwd_dqkwg(q, k, v, do):
    chunk_bwd_kernel_dqkwg[(1,)](q=q)
'''


class PlanWorkspacesFromKnowledgeTests(unittest.TestCase):
    def test_plan_emits_kernel_named_workspaces_per_launch(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            source_dir = repo / "src/kernels/fla/ops/common"
            source_dir.mkdir(parents=True)
            (source_dir / "chunk_o.py").write_text(SAMPLE_SOURCE, encoding="utf-8")
            knowledge = root / "PERF_KNOWLEDGE_BASE.md"
            knowledge.write_text(SAMPLE_KB, encoding="utf-8")
            output = root / "workspace-plan.json"

            code = main(
                [
                    "--knowledge",
                    knowledge.as_posix(),
                    "--repo",
                    repo.as_posix(),
                    "--output",
                    output.as_posix(),
                ],
            )
            self.assertEqual(code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))

        names = {entry["workspace"] for entry in payload["workspaces"]}
        self.assertEqual(
            names,
            {"chunk_bwd_kernel_dv_local", "chunk_bwd_kernel_dqkwg"},
        )
        dv_entry = next(
            item for item in payload["workspaces"] if item["workspace"] == "chunk_bwd_kernel_dv_local"
        )
        self.assertEqual(dv_entry["launch_functions"], ["chunk_bwd_dv_local"])
        self.assertEqual(dv_entry["operator_filename"], "chunk_bwd_kernel_dv_local.py")
        self.assertIn("85171374766b", dv_entry["knowledge_lessons"])


if __name__ == "__main__":
    unittest.main()
