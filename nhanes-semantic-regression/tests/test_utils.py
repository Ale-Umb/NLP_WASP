from __future__ import annotations

import unittest

import numpy as np

from nhanes_semantic.preprocess import normalize_values
from nhanes_semantic.utils import assign_partition


class UtilityTest(unittest.TestCase):
    def test_partition_is_stable_and_complete(self) -> None:
        values = np.arange(1000)
        fractions = {"factory": 0.2, "train": 0.5, "test": 0.3}
        first = assign_partition(values, fractions, 42, "rows")
        second = assign_partition(values, fractions, 42, "rows")
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(first), set(fractions))

    def test_normalization_imputes_and_clips(self) -> None:
        stat = {"q_low": -2.0, "q_high": 2.0, "median": 0.5, "mean": 0.0, "scale": 2.0}
        result = normalize_values(np.asarray([-10.0, np.nan, 10.0]), stat)
        np.testing.assert_allclose(result, [-1.0, 0.25, 1.0])


if __name__ == "__main__":
    unittest.main()

