import unittest
import yaml
import os
import shutil
from ..utils.test import get_np_dummy_data
from ..preprocessor import PCA, NMF
from ..unsupervised import GMM, KMeans, MeanShift
from ..transformation import Reflectance, TorchTransformation, TorchVisionTransformation
from ..supervised import SVM, QDA, LDA
from ..tv_transforms import Bandpass


TYPES_TO_CHECK = (int, float, str, bool, list, tuple)
TEST_DIR = "./test/temp"


class TestNodeSerialization():

    def test_serialization(self):
        os.makedirs(TEST_DIR, exist_ok=True)

        if self.node.serialize.__code__.co_argcount == 2:
            node_yaml = self.node.serialize(TEST_DIR)
        else:
            node_yaml = self.node.serialize()

        node_dict = yaml.full_load(node_yaml)
        lnode = self.node.__class__()

        if lnode.load.__code__.co_argcount == 3:
            lnode.load(params=node_dict, filepath=TEST_DIR)
        else:
            lnode.load(params=node_dict)

        load_ok = all((getattr(lnode, attr) == getattr(self.node, attr)
                       for attr in lnode.__dict__.keys()
                       if type(getattr(lnode, attr)) in TYPES_TO_CHECK))

        self.assertTrue(load_ok)
        shutil.rmtree(TEST_DIR)


class TestPreprocessorPCA(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = PCA(15)
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestPreprocessorNMF(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = NMF(15)
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestUnsupervisedKMeans(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = KMeans(15)
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestUnsupervisedGMM(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = GMM(15)
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestUnsupervisedMeanShift(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = MeanShift()
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestTransformationTorch(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = TorchTransformation("add", operand_b=5)
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestTransformationTorchVision(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = TorchVisionTransformation(Bandpass(5, 10))
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestTransformationReflectance(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = Reflectance(0.1, 1.8)
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)))


class TestSupervisedSVM(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = SVM()
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)),
                      get_np_dummy_data((10, 15, 20, 1)))


class TestSupervisedQDA(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = QDA()
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)),
                      get_np_dummy_data((10, 15, 20, 1)))


class TestSupervisedLDA(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = LDA()
        self.node.fit(get_np_dummy_data((10, 15, 20, 25)),
                      get_np_dummy_data((10, 15, 20, 1)))
