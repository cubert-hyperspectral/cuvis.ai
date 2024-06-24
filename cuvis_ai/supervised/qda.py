from .base_supervised import BaseSupervised
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as sk_qda

from ..utils.numpy_utils import flatten_batch_and_spatial, flatten_batch_and_labels, unflatten_batch_and_spatial, get_shape_without_batch

import numpy as np
import yaml
import numpy as np
import pickle as pk
import os

from dataclasses import dataclass


@dataclass
class QDA(BaseSupervised):

    def __post_init__(self):
        self.initialized = False

        self.qda = sk_qda()

    @BaseSupervised.input_dim.getter
    def input_dim(self):
        return self._input_dim

    @BaseSupervised.output_dim.getter
    def output_dim(self):
        return self._output_dim

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._input_dim = get_shape_without_batch(X, ignore=[0, 1])
        self._output_dim = (-1, -1, self.n_components)

        flatten_image = flatten_batch_and_spatial(X)
        flatten_l = flatten_batch_and_labels(Y)

        self.qda.fit(flatten_image, flatten_l)

        self.initialized = True

    def forward(self, X: np.ndarray):
        flatten_image = flatten_batch_and_spatial(X)

        predictions = self.qda.transform(flatten_image)
        predictions = unflatten_batch_and_spatial(predictions, X.shape)
        return predictions

    def serialize(self, serial_dir: str):
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.qda, open(os.path.join(
            serial_dir, f"{hash(self.qda)}_qda.pkl"), "wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'input_size': self.input_size,
            'qda_object': f"{hash(self.qda)}_qda.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str):
        self.id = params.get('id')
        self.input_size = params.get('input_size')
        self.qda = pk.load(
            open(os.path.join(filepath, params.get('qda_object')), 'rb'))
        self.initialized = True
