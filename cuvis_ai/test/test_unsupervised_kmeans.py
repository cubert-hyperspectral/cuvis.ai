import unittest

from ..unsupervised import KMeans
from ..utils.test import get_np_dummy_data


class TestUnsupervisedKMeans(unittest.TestCase):

    def setUp(self):
        self.node = KMeans(15)
        self.node.fit(get_np_dummy_data((10, 125, 100, 150)))

    def test_initialization(self):
        self.assertTrue(self.node.initialized)

    def test_correct_input_dim(self):
        data = get_np_dummy_data((125,))
        self.assertTrue(self.node.check_input_dim((125, 100, 150)))

    def test_incorrect_input_dim(self):
        self.assertTrue(self.node.check_output_dim((125, 100, 1)))

    def test_passthrough(self):
        pass


if __name__ == '__main__':
    unittest.main()
