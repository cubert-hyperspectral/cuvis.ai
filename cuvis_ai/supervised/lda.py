from ..node.base import BaseSupervised
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sk_lda

from ..node import Node
from ..utils.numpy_utils import flatten_batch_and_spatial, flatten_batch_and_labels, unflatten_batch_and_spatial, get_shape_without_batch
import yaml
import numpy as np
import pickle as pk
import os

from dataclasses import dataclass


@dataclass
class LDA(Node, BaseSupervised):
    solver: str = 'svd'
    n_components: int = None

    def __post_init__(self):
        super().__init__()
        self.initialized = False
        self.input_size = (-1, -1, -1)
        self.output_size = (-1, -1, -1)
        self.lda = sk_lda(solver=self.solver, n_components=self.n_components)

    @Node.input_dim.getter
    def input_dim(self):
        return self.input_size

    @Node.output_dim.getter
    def output_dim(self):
        return self.output_size

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.input_size = get_shape_without_batch(X, ignore=[0, 1])
        self.output_size = (-1, -1, self.n_components)

        flatten_image = flatten_batch_and_spatial(X)
        flatten_l = flatten_batch_and_labels(Y)

        self.lda.fit(flatten_image, flatten_l)

        self.initialized = True

    def forward(self, X: np.ndarray):
        flatten_image = flatten_batch_and_spatial(X)

        predictions = self.lda.transform(flatten_image)
        predictions = unflatten_batch_and_spatial(predictions, X.shape)
        return predictions

    def serialize(self, serial_dir: str):
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.lda, open(os.path.join(
            serial_dir, f"{hash(self.lda)}_lda.pkl"), "wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'n_components': self.n_components,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'lda_object': f"{hash(self.lda)}_lda.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str):
        self.id = params.get('id')
        self.input_size = params.get('input_size')
        self.output_size = params.get('output_size')
        self.n_components = params.get('n_components')
        self.lda = pk.load(
            open(os.path.join(filepath, params.get('lda_object')), 'rb'))
        self.initialized = True
