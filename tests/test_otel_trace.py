from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.otel_trace import new_trace_run_id


class RunIdTests(unittest.TestCase):
    def test_new_trace_run_id_uses_second_precision_on_first_allocation(self) -> None:
        fixed = datetime(2026, 6, 23, 9, 29, 59, 505637, tzinfo=timezone.utc)

        with patch("triton_agent.otel_trace._RUN_ID_COLLISION_COUNTS", Counter()), patch(
            "triton_agent.otel_trace.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = fixed

            run_id = new_trace_run_id(prefix="optimize")

        self.assertEqual(run_id, "optimize-20260623-092959")

    def test_new_trace_run_id_appends_numeric_suffix_for_same_second_collision(self) -> None:
        fixed = datetime(2026, 6, 23, 9, 29, 59, 505637, tzinfo=timezone.utc)

        with patch("triton_agent.otel_trace._RUN_ID_COLLISION_COUNTS", Counter()), patch(
            "triton_agent.otel_trace.datetime"
        ) as mock_datetime:
            mock_datetime.now.side_effect = [fixed, fixed, fixed]

            first = new_trace_run_id(prefix="optimize")
            second = new_trace_run_id(prefix="optimize")
            third = new_trace_run_id(prefix="optimize")

        self.assertEqual(first, "optimize-20260623-092959")
        self.assertEqual(second, "optimize-20260623-092959-2")
        self.assertEqual(third, "optimize-20260623-092959-3")

    def test_new_trace_run_id_without_prefix_uses_same_collision_rule(self) -> None:
        fixed = datetime(2026, 6, 23, 9, 29, 59, 505637, tzinfo=timezone.utc)

        with patch("triton_agent.otel_trace._RUN_ID_COLLISION_COUNTS", Counter()), patch(
            "triton_agent.otel_trace.datetime"
        ) as mock_datetime:
            mock_datetime.now.side_effect = [fixed, fixed]

            first = new_trace_run_id()
            second = new_trace_run_id()

        self.assertEqual(first, "20260623-092959")
        self.assertEqual(second, "20260623-092959-2")
