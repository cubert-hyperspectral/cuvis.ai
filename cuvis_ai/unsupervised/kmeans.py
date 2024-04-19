import os
import yaml
import numpy as np
import uuid
import pickle as pk
import matplotlib.pyplot as plt
from ..node import Node
from .base_unsupervised import BaseUnsupervised
from sklearn.cluster import KMeans as sk_kmeans

class KMeans(Node, BaseUnsupervised):
    """
    K-Means based unsupervised classifier
    """
    
    def __init__(self, n_clusters: int=None):
        self.n_clusters = n_clusters
        self.input_size = None
        self.initialized = False
        self.id =  f'{self.__class__.__name__}-{str(uuid.uuid4())}'

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
        self.input_size = (-1,-1,X.shape[2]) # Constrain the number of wavelengths or input features
        self.output_size = (-1,-1,1)
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
            'id': self.id,
            'n_clusters': self.n_clusters,
            'input_size': self.input_size,
            'kmeans_object': f"{hash(self.fit_kmeans)}_kmeans.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str):
        '''
        Load dumped parameters to recreate the K-Means object
        '''
        self.id = params.get('id')
        self.input_size = params.get('input_size')
        self.n_clusters = params.get('n_clusters')
        self.fit_kmeans = pk.load(open(os.path.join(filepath, params.get('kmeans_object')),'rb'))
        self.initialized = True