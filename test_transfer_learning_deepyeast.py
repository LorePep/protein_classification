import csv
import os
import unittest

import numpy as np
from PIL import Image
from numpy.testing import assert_array_almost_equal

from transfer_learning_deepyeast import (
    load_dataset_rg,
    _parse_label,
    _get_images_ids_to_paths,
    _get_images_ids_to_labels)

FIXTURE_PATH = "testdata"
FIXTURE_IMG_PATH = "testdata/80ec1c96-bba8-11e8-b2ba-ac1f6b6435d0.png"
FIXTURE_LABELS_PATH = "testdata/train.csv"


class TestTransferLearning(unittest.TestCase):

    # def test_loda_dataset_rg(self):
    #     expected = _get_fixture_img(FIXTURE_IMG_PATH)
        
    #     actual = load_dataset_rg(FIXTURE_PATH)
    #     # 1 image * 512 w * 512 h * 2 channels
    #     self.assertEqual(actual.size, 524288)
    #     assert_array_almost_equal(actual[0, :, :, 0], expected[:, :, 0])
    #     assert_array_almost_equal(actual[0, :, :, 1], expected[:, :, 1])

    def test_get_images_ids_to_labels(self):
        input_label_file = csv.DictReader(open(FIXTURE_LABELS_PATH))
        actual = _get_images_ids_to_labels(input_label_file)

        first_expected_label = np.zeros((1, 28))
        first_expected_label[0, 16] = 1
        first_expected_label[0, 0] = 1

        second_expected_label = np.zeros((1, 28))
        second_expected_label[0, 7] = 1
        second_expected_label[0, 1] = 1

        expected = {
            "firstID": first_expected_label,
            "secondID": second_expected_label,
        }

        self.assertEqual(len(expected), len(actual))
        for k, value in expected.items():
            self.assertIn(k, actual)
            assert_array_almost_equal(value, actual[k])
    
    def test_get_images_ids_to_paths(self):
        paths = ["foo/oneID.png", "bar/anotherID.png"]
        actual = _get_images_ids_to_paths(paths)

        expected = {
            "oneID": "foo/oneID.png",
            "anotherID": "bar/anotherID.png"
        }

        self.assertEqual(expected, actual)

    def test_get_images_ids_to_paths_bad(self):
        paths = ["foo/badpath"]
        with self.assertRaises(ValueError):
            _get_images_ids_to_paths(paths)
    
    def test_parse_label(self):
        actual = _parse_label("2 0 1")
        expected = np.zeros((1, 28))
        expected[0, 0] = 1
        expected[0, 1] = 1
        expected[0, 2] = 1
        assert_array_almost_equal(expected, actual)


def _get_fixture_img(img_path):
    img = Image.open(img_path)
    img.load()
    data = np.asarray(img, dtype=np.uint8)

    return data
