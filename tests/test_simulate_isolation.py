import json
import tempfile
import unittest
from pathlib import Path

from triton_agent.pattern_validation_loop.simulate_isolation import isolate_workspace_for_simulate

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SimulateIsolationTests(unittest.TestCase):
    def test_isolate_hides_and_restores_validation_meta(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT / "tests") as tmp:
            workspace = Path(tmp)
            meta = workspace / "validation-meta.json"
            meta.write_text(
                json.dumps({"expected_patterns": ["secret-pattern"]}) + "\n",
                encoding="utf-8",
            )
            with isolate_workspace_for_simulate(workspace):
                self.assertFalse(meta.is_file())
                held = workspace / ".triton-agent/offline-eval-held/validation-meta.json"
                self.assertTrue(held.is_file())
            self.assertTrue(meta.is_file())
            self.assertFalse((workspace / ".triton-agent/offline-eval-held").exists())


if __name__ == "__main__":
    unittest.main()
