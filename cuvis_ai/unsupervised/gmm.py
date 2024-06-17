import os
import yaml
import uuid
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from ..node import Node
from .base_unsupervised import BaseUnsupervised
from sklearn.mixture import GaussianMixture as sk_gmm


class GMM(Node, BaseUnsupervised):
    """Gaussian Mixture Model Classifier

    Parameters
    ----------
    Node : Abstract Node, shared by all CUVIS.AI classes
    BaseUnsupervised : Secondary inheritance for unsupervised nodes 
    """

    def __init__(self, n_clusters: int = None):
        """Initialize a GMM unsupervised classifier

        Parameters
        ----------
        n_clusters : int, optional
            number of clusters to seed for GMM clustering, by default None
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.input_size = None
        self.initialized = False
        self.input_size = (-1, -1, -1)
        self.output_size = (-1, -1, -1)

    def fit(self, X: np.ndarray):
        """Train the GMM classifier given a sample datacube.

        Parameters
        ----------
        X : np.ndarray
            Training data for classifier in shape of W x H x C
        """
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        self.fit_gmm = sk_gmm(n_components=self.n_clusters)
        self.fit_gmm.fit(image_2d)
        # Set the dimensions for a later check
        # Constrain the number of wavelengths or input features
        self.input_size = (-1, -1, X.shape[2])
        self.output_size = (-1, -1, 1)
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
        """Apply GMM classifier to new data

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
        data = self.fit_gmm.predict(image_2d)
        cube_data = data.reshape((X.shape[0], X.shape[1]))
        return cube_data

    def serialize(self, serial_dir: str) -> str:
        """Write the model parameters to a YAML format and save GMMs weights

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
        pk.dump(self.fit_gmm, open(os.path.join(
            serial_dir, f"{hash(self.fit_gmm)}_gmm.pkl"), "wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'n_clusters': self.n_clusters,
            'input_size': self.input_size,
            'gmm_object': f"{hash(self.fit_gmm)}_gmm.pkl"
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
        self.n_clusters = params.get('n_clusters')
        self.fit_gmm = pk.load(
            open(os.path.join(filepath, params.get('gmm_object')), 'rb'))
        self.initialized = True
