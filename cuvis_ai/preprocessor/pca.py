
import pickle as pk
import numpy as np
from ..node import Node
from ..utils.numpy import flatten_batch_and_spatial, unflatten_batch_and_spatial

from ..node.base import Preprocessor
from sklearn.decomposition import PCA as sk_pca
from pathlib import Path


class PCA(Node, Preprocessor):
    """
    Principal Component Analysis (PCA) preprocessor.
    """

    def __init__(self, n_components: int = None):
        super().__init__()
        self.n_components = n_components
        self.initialized = False
        self.input_size = (-1, -1, -1)
        self.output_size = (-1, -1, self.n_components)

    def fit(self, X: np.ndarray):
        """
        Fit PCA to the data.

        Parameters:
        X (array-like): Input data.

        Returns:
        self
        """
        image_2d = flatten_batch_and_spatial(X)
        self.fit_pca = sk_pca(n_components=self.n_components)
        self.fit_pca.fit(image_2d)
        # Set the dimensions for a later check
        # Constrain the number of wavelengths
        self.input_size = (-1, -1, image_2d.shape[1])
        self.output_size = (-1, -1, self.n_components)
        # Initialization is complete
        self.initialized = True

    def forward(self, X: np.ndarray):
        """
        Transform the input data.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        # Transform data using precomputed PCA components
        image_2d = flatten_batch_and_spatial(X)
        data = self.fit_pca.transform(image_2d)
        return unflatten_batch_and_spatial(data, X.shape)

    @Node.input_dim.getter
    def input_dim(self):
        return self.input_size

    @Node.output_dim.getter
    def output_dim(self):
        return self.output_size

    def serialize(self, serial_dir: str) -> dict:
        '''
        This method should dump parameters to a yaml file format
        '''
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        with open(Path(serial_dir) / f"{hash(self.fit_pca)}_pca.pkl", "wb") as f:
            pk.dump(self.fit_pca, f)

        data = {
            'id': self.id,
            'n_components': self.n_components,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'pca_object': f"{hash(self.fit_pca)}_pca.pkl"
        }
        return data

    def load(self, params: dict, serial_dir: str):
        '''
        Load dumped parameters to recreate the pca object
        '''
        self.id = params.get('id')
        self.input_size = tuple(params.get('input_size'))
        self.n_components = params.get('n_components')
        self.output_size = tuple(params.get('output_size'))
        with open(Path(serial_dir) / params.get('pca_object'), 'rb') as f:
            self.fit_pca = pk.load(f)
        self.initialized = True
