import unittest

from ..utils.numpy import *
from ..utils.test import get_np_dummy_data


class TestNumpyUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_shape_without_batch(self):

        data = get_np_dummy_data((4, 3, 2, 1))

        self.assertTrue((3, 2, 1) == get_shape_without_batch(data))

        self.assertTrue(
            (3, 2, -1) == get_shape_without_batch(data, ignore=[2]))

        self.assertTrue(
            (3, 2, -1) == get_shape_without_batch(data, ignore=(2)))

        self.assertTrue(
            (3, 2, -1) == get_shape_without_batch(data, ignore=(2,)))

        self.assertTrue(
            (3, -1, -1) == get_shape_without_batch(data, ignore=(2, 1)))


if __name__ == '__main__':
    unittest.main()
