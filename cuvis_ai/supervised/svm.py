from .base_supervised import BaseSupervised
from sklearn import svm as sk_svm

from ..node import Node
from ..utils.numpy_utils import flatten_batch_and_spatial, flatten_batch_and_labels, unflatten_batch_and_spatial, get_shape_without_batch

import numpy as np
import yaml
import numpy as np
import pickle as pk
import os


class SVM(Node, BaseSupervised):

    def __init__(self) -> None:
        super().__init__()

        self.svm = sk_svm.SVC()
        self.input_size = (-1, -1, -1)
        self.output_size = (-1, -1, -1)

    @Node.input_dim.getter
    def input_dim(self):
        return self.input_size

    @Node.output_dim.getter
    def output_dim(self):
        return self.output_size

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.input_size = get_shape_without_batch(X, ignore=[0, 1])

        flatten_image = flatten_batch_and_spatial(X)
        flatten_l = flatten_batch_and_labels(Y)

        self.svm.fit(flatten_image, flatten_l)

        self.initialized = True

    def forward(self, X: np.ndarray):
        flatten_image = flatten_batch_and_spatial(X)

        predictions = self.svm.predict(flatten_image)
        predictions = unflatten_batch_and_spatial(predictions, X.shape)
        return predictions

    def serialize(self, serial_dir: str):
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.svm, open(os.path.join(
            serial_dir, f"{hash(self.svm)}_svm.pkl"), "wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'svm_object': f"{hash(self.svm)}_svm.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str):
        self.id = params.get('id')
        self.input_size = params.get('input_size')
        self.output_size = params.get('output_size')
        self.svm = pk.load(
            open(os.path.join(filepath, params.get('svm_object')), 'rb'))
        self.initialized = True
