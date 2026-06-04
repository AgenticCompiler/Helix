import sys
import unittest
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-pattern-validation-loop"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

from reference_test import (  # noqa: E402
    is_reference_test_filename,
    reference_test_destination_name,
)


class ReferenceTestNamingTests(unittest.TestCase):
    def test_maps_py_to_py_txt(self) -> None:
        self.assertEqual(
            reference_test_destination_name("test_chunk_o.py"),
            "test_chunk_o.py.txt",
        )

    def test_idempotent_when_already_txt(self) -> None:
        self.assertEqual(
            reference_test_destination_name("test_chunk_o.py.txt"),
            "test_chunk_o.py.txt",
        )

    def test_is_reference_test_filename(self) -> None:
        self.assertTrue(is_reference_test_filename("test_foo.py.txt"))
        self.assertFalse(is_reference_test_filename("test_foo.py"))


if __name__ == "__main__":
    unittest.main()
