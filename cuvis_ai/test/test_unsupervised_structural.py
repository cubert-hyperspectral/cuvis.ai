import unittest

from ..unsupervised import KMeans, GMM, MeanShift
from ..utils.test import get_np_dummy_data

from functools import wraps


class TestUnsupervisedNode():

    def setUp(self):
        # setup is handled by the decorator
        pass

    def test_initialization(self):
        self.assertTrue(self.node.initialized)
        self.assertTrue(self.node.input_dim)
        self.assertTrue(self.node.output_dim)

    def test_correct_input_dim(self):
        self.assertTrue(self.node.check_input_dim((125, 100, 150)))

        data = get_np_dummy_data((10, 125, 100, 150))
        self.assertTrue(self.node.check_input_dim(data))

        data = get_np_dummy_data((125, 100, 150))
        self.assertTrue(self.node.check_input_dim(data))

    def test_incorrect_input_dim(self):
        self.assertTrue(self.node.check_output_dim((125, 100, 1)))

    def test_passthrough(self):

        # check if passthrough generates the correct shape
        data = get_np_dummy_data((10, 125, 100, 150))

        output = self.node.forward(data)

        self.assertTrue(output.shape == (10, 125, 100, 1))

        data = get_np_dummy_data((125, 100, 150))

        output = self.node.forward(data)

        self.assertTrue(output.shape == (125, 100, 1))


class TestUnsupervisedKMeans(TestUnsupervisedNode, unittest.TestCase):

    def setUp(self):
        self.node = KMeans(15)
        self.node.fit(get_np_dummy_data((10, 125, 100, 150)))


# class TestUnsupervisedGMM(TestUnsupervisedNode, unittest.TestCase):
#
#    def setUp(self):
#        self.node = GMM(15)
#        self.node.fit(get_np_dummy_data((10, 125, 100, 150)))


# class TestUnsupervisedMeanShift(TestUnsupervisedNode, unittest.TestCase):
#
#    def setUp(self):
#        self.node = MeanShift()
#        self.node.fit(get_np_dummy_data((10, 125, 100, 150)))


if __name__ == '__main__':
    unittest.main()
