import os
import yaml
import pickle as pk
import numpy as np
import uuid
from ..node import Node
from .base_preprocessor import Preprocessor
from sklearn.decomposition import NMF as sk_nmf

class NMF(Node, Preprocessor):
    """
    Non-Negative Matrix Factorization (NMF) preprocessor.
    """
    
    def __init__(self, n_components: int=None):
        super().__init__()
        self.n_components = n_components
        self.input_size = None
        self.output_size = None
        self.initialized = False
        self.id =  f'{self.__class__.__name__}-{str(uuid.uuid4())}'

    def fit(self, X: np.ndarray):
        """
        Fit NMF to the data.

        Parameters:
        X (array-like): Input data.

        Returns:
        self
        """
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        self.fit_nmf = sk_nmf(n_components=self.n_components)
        self.fit_nmf.fit(image_2d)
        # Set the dimensions for a later check
        self.input_size = X.shape[2] # Constrain the number of wavelengths
        self.output_size = self.n_components
        # Initialization is complete
        self.initialized = True

    def check_input_dim(self, X: np.ndarray):
        assert(X.shape[2] == self.input_size)

    def check_output_dim(self, X: np.ndarray):
        assert(X.shape[2] == self.n_components)
    
    def forward(self, X: np.ndarray):
        """
        Transform the input data.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        # Transform data using precomputed NMF components
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        data = self.fit_nmf.transform(image_2d)
        cube_data = data.reshape((X.shape[0], X.shape[1], self.n_components))
        return cube_data

    def serialize(self, serial_dir: str):
        '''
        This method should dump parameters to a yaml file format
        '''
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.fit_nmf, open(os.path.join(serial_dir,f"{hash(self.fit_nmf)}_nmf.pkl"),"wb"))
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
        self.fit_nmf = pk.load(open(os.path.join(filepath,params.get('nmf_object')),'rb'))
        self.initialized = True

