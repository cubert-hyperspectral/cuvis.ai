import unittest
import yaml
import os
import shutil
import numpy as np
from ..utils.test import get_np_dummy_data
from ..preprocessor import PCA, NMF
from ..unsupervised import GMM, KMeans, MeanShift
from ..transformation import Reflectance, TorchTransformation, TorchVisionTransformation
from ..supervised import SVM, QDA, LDA
from ..tv_transforms import Bandpass
from ..utils.serializer import YamlSerializer


TYPES_TO_CHECK = (int, float, str, bool, list, tuple, np.ndarray)
TEST_DIR = "./test/temp"


class TestNodeSerialization():

    def test_serialization(self):
        os.makedirs(TEST_DIR, exist_ok=True)

        node_params = self.node.serialize(TEST_DIR)

        serializer = YamlSerializer(TEST_DIR, 'test_node')
        serializer.serialize(node_params)

        node_dict = serializer.load()

        lnode = self.node.__class__()

        lnode.load(node_dict, TEST_DIR)

        load_ok = True
        for attr in lnode.__dict__.keys():
            if type(getattr(lnode, attr)) not in TYPES_TO_CHECK:
                continue
            if getattr(lnode, attr) != getattr(self.node, attr):
                print(F"Attribute '{attr}' not equal! {
                      getattr(lnode, attr)} != {getattr(self.node, attr)}")
                load_ok = False
        shutil.rmtree(TEST_DIR)

        self.assertTrue(load_ok)


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
        self.node.fit(get_np_dummy_data((15, 20, 25)),
                      np.where(get_np_dummy_data((15, 20, 1)) > 0.5, 1, 0))


class TestSupervisedQDA(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = QDA()
        self.node.fit(get_np_dummy_data((15, 20, 25)),
                      np.where(get_np_dummy_data((15, 20, 1)) > 0.5, 1, 0))


class TestSupervisedLDA(TestNodeSerialization, unittest.TestCase):

    def setUp(self):
        self.node = LDA()
        self.node.fit(get_np_dummy_data((15, 20, 25)),
                      np.where(get_np_dummy_data((15, 20, 1)) > 0.5, 1, 0))
