import os
import yaml
import pickle as pk
import numpy as np
from ..node import Node
from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial
from ..node.base import Preprocessor
from sklearn.decomposition import NMF as sk_nmf


class NMF(Node, Preprocessor):
    """
    Non-Negative Matrix Factorization (NMF) preprocessor.
    """

    def __init__(self, n_components: int = None):
        super().__init__()
        self.n_components = n_components
        self.input_size = (-1, -1, -1)
        self.output_size = (-1, -1, -1)
        self.initialized = False

    def fit(self, X: np.ndarray):
        """
        Fit NMF to the data.

        Parameters:
        X (array-like): Input data.

        Returns:
        self
        """
        image_2d = flatten_batch_and_spatial(X)
        self.fit_nmf = sk_nmf(n_components=self.n_components)
        self.fit_nmf.fit(image_2d)
        # Set the dimensions for a later check
        # Constrain the number of wavelengths
        self.input_size = (-1, -1, image_2d.shape[1])
        self.output_size = (-1, -1, self.n_components)
        # Initialization is complete
        self.initialized = True

    @Node.input_dim.getter
    def input_dim(self):
        return self.input_size

    @Node.output_dim.getter
    def output_dim(self):
        return self.output_size

    def forward(self, X: np.ndarray):
        """
        Transform the input data.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        # Transform data using precomputed NMF components
        image_2d = flatten_batch_and_spatial(X)
        data = self.fit_nmf.transform(image_2d)
        return unflatten_batch_and_spatial(data, X.shape)

    def serialize(self, serial_dir: str) -> str:
        '''
        This method should dump parameters to a yaml file format
        '''
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.fit_nmf, open(os.path.join(
            serial_dir, f"{hash(self.fit_nmf)}_nmf.pkl"), "wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'n_components': self.n_components,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'nmf_object': f"{hash(self.fit_nmf)}_nmf.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str):
        '''
        Load dumped parameters to recreate the nmf object
        '''
        self.id = params.get('id')
        self.input_size = params.get('input_size')
        self.n_components = params.get('n_components')
        self.output_size = params.get('output_size')
        self.fit_nmf = pk.load(
            open(os.path.join(filepath, params.get('nmf_object')), 'rb'))
        self.initialized = True
