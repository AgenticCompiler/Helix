import unittest

from triton_agent.pattern_validation_loop.orchestration import (
    _build_pattern_validation_optimize_prompt,
)
from triton_agent.pattern_validation_loop.reference_tests import (
    REFERENCE_TEST_SUFFIX,
    build_pattern_validation_optimize_reference_test_prompt,
)


class PatternValidationReferenceTestTests(unittest.TestCase):
    def test_reference_test_suffix_constant(self) -> None:
        self.assertEqual(REFERENCE_TEST_SUFFIX, ".py.txt")

    def test_optimize_prompt_mentions_reference_tests(self) -> None:
        prompt = build_pattern_validation_optimize_reference_test_prompt()
        self.assertIn(".py.txt", prompt)
        self.assertIn("dtype", prompt)
        self.assertIn("shape", prompt)

    def test_build_optimize_prompt_appends_user_instructions(self) -> None:
        combined = _build_pattern_validation_optimize_prompt("Use pattern compile_hint.")
        self.assertIn(".py.txt", combined)
        self.assertIn("Use pattern compile_hint.", combined)


if __name__ == "__main__":
    unittest.main()
