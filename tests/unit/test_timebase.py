from __future__ import annotations

import unittest

from volleyball_vision.timebase import (
    fit_affine_clock_mapping,
    integer_midpoint_ns,
    map_timestamp_ns,
)


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

    def test_affine_mapping_uses_centered_epoch_arithmetic(self) -> None:
        mapping = fit_affine_clock_mapping(
            [1_800_000_000_000_000_000, 1_800_000_000_010_000_000],
            [2_000_000_000, 2_010_001_000],
            source_clock="source",
            reference_clock="reference",
            mapping_id="mapping-1",
            source="fixture",
        )

        self.assertIsNotNone(mapping)
        assert mapping is not None
        self.assertEqual(
            map_timestamp_ns(1_800_000_000_005_000_000, mapping),
            2_005_000_500,
        )


if __name__ == "__main__":
    unittest.main()
