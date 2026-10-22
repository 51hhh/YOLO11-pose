from __future__ import annotations

import unittest

from volleyball_vision.timebase import integer_midpoint_ns


class TimebaseTests(unittest.TestCase):
    def test_epoch_sized_odd_midpoint_uses_exact_integer_arithmetic(self) -> None:
        left_ns = 1_800_000_000_000_000_001
        right_ns = 1_800_000_000_000_000_002

        self.assertEqual(
            integer_midpoint_ns(left_ns, right_ns),
            1_800_000_000_000_000_001,
        )

    def test_negative_timestamp_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "非负整数"):
            integer_midpoint_ns(-1, 0)


if __name__ == "__main__":
    unittest.main()
