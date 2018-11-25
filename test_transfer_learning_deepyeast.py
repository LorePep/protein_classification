import os
import unittest

import numpy as np
from PIL import Image
from numpy.testing import assert_array_almost_equal

from transfer_learning_deepyeast import load_dataset_rg

FIXTURE_PATH = "testdata"
FIXTURE_IMG_PATH = "testdata/80ec1c96-bba8-11e8-b2ba-ac1f6b6435d0.png"


class TestTransferLearning(unittest.TestCase):

    def test_loda_dataset_rg(self):
        expected = _get_fixture_img(FIXTURE_IMG_PATH)
        
        actual = load_dataset_rg(FIXTURE_PATH)
        # 1 image * 512 w * 512 h * 2 channels
        self.assertEqual(actual.size, 524288)
        assert_array_almost_equal(actual[0, :, :, 0], expected[:, :, 0])
        assert_array_almost_equal(actual[0, :, :, 1], expected[:, :, 1])


def _get_fixture_img(img_path):
    img = Image.open(img_path)
    img.load()
    data = np.asarray(img, dtype=np.uint8)

    return data
