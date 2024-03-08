import os
import yaml
import numpy as np
import typing
from typing import Dict
import pickle as pk
import matplotlib.pyplot as plt
from .base_unsupervised import BaseUnsupervised
from sklearn.cluster import KMeans as sk_kmeans

class KMeans(BaseUnsupervised):
    """
    K-Means based unsupervised classifier
    """
    
    def __init__(self, n_clusters: int=None):
        self.n_clusters = n_clusters
        self.input_size = None
        self.initialized = False
        
    def fit(self, X: np.ndarray):
        """
        Fit K-Means to the data.

        Parameters:
        X (array-like): Input data.

        Returns:
        self
        """
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        self.fit_kmeans = sk_kmeans(n_clusters=self.n_clusters)
        self.fit_kmeans.fit(image_2d)
        # Set the dimensions for a later check
        self.input_size = X.shape[2] # Constrain the number of wavelengths or input features
        # Initialization is complete
        self.initialized = True

    def check_input_dim(self, X: np.ndarray):
        assert(X.shape[2] == self.input_size)
    
    def forward(self, X: np.ndarray):
        """
        Transform the input data.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        # Transform data using precomputed K-Means components
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        data = self.fit_kmeans.predict(image_2d)
        cube_data = data.reshape((X.shape[0], X.shape[1]))
        return cube_data

    def serialize(self, serial_dir: str):
        '''
        This method should dump parameters to a yaml file format
        '''
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.fit_kmeans, open(os.path.join(serial_dir,f"{hash(self.fit_kmeans)}_kmeans.pkl"),"wb"))
        data = {
            'type': type(self).__name__,
            'n_clusters': self.n_clusters,
            'input_size': self.input_size,
            'kmeans_object': f"{hash(self.fit_kmeans)}_kmeans.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: Dict, filepath: str):
        '''
        Load dumped parameters to recreate the K-Means object
        '''
        self.input_size = params.get('input_size')
        self.n_clusters = params.get('n_clusters')
        self.fit_kmeans = pk.load(open(os.path.join(filepath, params.get('kmeans_object')),'rb'))
        self.initialized = True