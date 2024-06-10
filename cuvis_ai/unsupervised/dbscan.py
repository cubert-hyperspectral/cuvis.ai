import os
import yaml
import uuid
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from ..node import Node
from typing import Union, Optional, Callable
from .base_unsupervised import BaseUnsupervised
from sklearn.cluster import DBSCAN as sk_dbscan

class DBSCAN(Node, BaseUnsupervised):
    """Density-based spatial clustering of applications with noise (DBSCAN)

    Parameters
    ----------
    Node : Abstract Node, shared by all CUVIS.AI classes
    BaseUnsupervised : Secondary inheritance for unsupervised nodes 
    """
    
    def __init__(self, algorithm: Optional[Union[callable, str]] = 'euclidean'):
        """Initialize a DBSCAN clustering algorithm.

        Parameters
        ----------
        algorithm : Optional[Union[callable, str]], optional
           Name of distance function to use in clustering metric or a callable function, by default euclidean distance
        """
        super().__init__()
        self.id = F"{self.__class__.__name__}-{str(uuid.uuid4())}"
        self.algorithm = algorithm
        self.input_size = None
        self.initialized = False
        self.input_size = (-1,-1,-1)
        self.output_size = (-1,-1,-1)

    def fit(self, X: np.ndarray):
        """Train the DBSCAN classifier given a sample datacube.

        Parameters
        ----------
        X : np.ndarray
            Training data for classifier in shape of W x H x C
        """
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        self.fit_dbscan = sk_dbscan(metric=self.algorithm)
        self.fit_dbscan.fit(image_2d)
        # Set the dimensions for a later check
        self.input_size = (-1,-1,X.shape[2]) # Constrain the number of wavelengths or input features
        self.output_size = (-1,-1,1)
        # Initialization is complete
        self.initialized = True

    @Node.input_dim.getter
    def input_dim(self) -> int:
        """Get required input dimension.

        Returns
        -------
        int
            Number of channels
        """
        return self.input_size
    
    @Node.output_dim.getter
    def output_dim(self) -> int:
        """Get required output dimension.

        Returns
        -------
        _type_
            _description_
        """
        return self.output_size
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply DBSCAN classifier to new data

        Parameters
        ----------
        X : np.ndarray
            Array in W x H x C defining a hyperspectral datacube

        Returns
        -------
        np.ndarray
            W x H class predictions
        """
        # Transform data using precomputed K-Means components
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        data = self.fit_dbscan.predict(image_2d)
        cube_data = data.reshape((X.shape[0], X.shape[1]))
        return cube_data

    def serialize(self, serial_dir: str) -> str:
        """Write the model parameters to a YAML format and save DBSCAN weights

        Parameters
        ----------
        serial_dir : str
            Path to where weights should be saved

        Returns
        -------
        str
            YAML formatted string which will can be safely written to file.
        """
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.fit_dbscan, open(os.path.join(serial_dir,f"{hash(self.fit_dbscan)}_dbscan.pkl"),"wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'algorithm': self.algorithm,
            'input_size': self.input_size,
            'dbscan_object': f"{hash(self.fit_dbscan)}_dbscan.pkl"
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str):
        """_summary_

        Parameters
        ----------
        params : dict
            Parameters loaded form YAML file 
        filepath : str
            Path to unzipped directory containing stored matrices and weights.
        """
        self.id = params.get('id')
        self.input_size = params.get('input_size')
        self.algorithm = params.get('algorithm')
        self.fit_dbscan = pk.load(open(os.path.join(filepath, params.get('dbscan_object')),'rb'))
        self.initialized = True