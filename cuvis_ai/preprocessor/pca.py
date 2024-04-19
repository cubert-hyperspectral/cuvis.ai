import os
import yaml
import pickle as pk
import numpy as np
import uuid
from ..node import Node
from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial

from .base_preprocessor import Preprocessor
from sklearn.decomposition import PCA as sk_pca

class PCA(Node, Preprocessor):
    """
    Principal Component Analysis (PCA) preprocessor.
    """
    
    def __init__(self, n_components: int=None):
        super().__init__()
        self.n_components = n_components
        self.initialized = False
        self.id =  f'{self.__class__.__name__}-{str(uuid.uuid4())}'
        self.input_size = (-1,-1,-1)
        self.output_size = (-1,-1,-1)

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
        self.input_size = (-1,-1,X.shape[2]) # Constrain the number of wavelengths
        self.output_size = (-1,-1,self.n_components)
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

    def serialize(self, serial_dir: str):
        '''
        This method should dump parameters to a yaml file format
        '''
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.fit_pca, open(os.path.join(serial_dir,f"{hash(self.fit_pca)}_pca.pkl"),"wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'n_components': self.n_components,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'pca_object': f"{hash(self.fit_pca)}_pca.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str):
        '''
        Load dumped parameters to recreate the pca object
        '''
        self.id = params.get('id')
        self.input_size = params.get('input_size')
        self.n_components = params.get('n_components')
        self.output_size = params.get('output_size')
        self.fit_pca = pk.load(open(os.path.join(filepath, params.get('pca_object')),'rb'))
        self.initialized = True

