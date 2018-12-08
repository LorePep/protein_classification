import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from analyse_results import _apply_thresholds


class TestAnalyseResults(unittest.TestCase):

    def test_apply_thresholds(self):
        arr = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 5, 6],
        ])

        th = np.array([3, 4, 5])

        actual = _apply_thresholds(arr, th)
        expected = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
        ])
        assert_array_almost_equal(expected, actual)

    def test_apply_thresholds_bad(self):
        arr = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])

        bad_th = np.array([3, 2, 3, 5])

        with self.assertRaises(ValueError):
            _apply_thresholds(arr, bad_th)


