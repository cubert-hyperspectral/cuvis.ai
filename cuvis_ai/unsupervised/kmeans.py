import os
import yaml
import uuid
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from ..node import Node
from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial
from ..node.base import BaseUnsupervised
from sklearn.cluster import KMeans as sk_kmeans


class KMeans(Node, BaseUnsupervised):
    """K-Means classifier

    Parameters
    ----------
    Node : Abstract Node, shared by all CUVIS.AI classes
    BaseUnsupervised : Secondary inheritance for unsupervised nodes 
    """

    def __init__(self, n_clusters: int = None):
        """Initialize a K-Means unsupervised classifier

        Parameters
        ----------
        n_clusters : int, optional
            number of clusters to seed for k-means clustering, by default None
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.input_size = None
        self.initialized = False
        self.input_size = (-1, -1, -1)
        self.output_size = (-1, -1, -1)

    def fit(self, X: np.ndarray):
        """Train the K-Means classifier given a sample datacube.

        Parameters
        ----------
        X : np.ndarray
            Training data for classifier in shape of W x H x C
        """
        image_2d = flatten_batch_and_spatial(X)
        self.fit_kmeans = sk_kmeans(n_clusters=self.n_clusters)
        self.fit_kmeans.fit(image_2d)
        # Set the dimensions for a later check
        # Constrain the number of wavelengths or input features
        self.input_size = (-1, -1, image_2d.shape[1])
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
        """Apply K-Mean classifier to new data

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
        image_2d = flatten_batch_and_spatial(X)
        data = self.fit_kmeans.predict(image_2d)
        return unflatten_batch_and_spatial(data, X.shape)

    def serialize(self, serial_dir: str) -> str:
        """Write the model parameters to a YAML format and save K-Means weights

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
        pk.dump(self.fit_kmeans, open(os.path.join(
            serial_dir, f"{hash(self.fit_kmeans)}_kmeans.pkl"), "wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'n_clusters': self.n_clusters,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'kmeans_object': f"{hash(self.fit_kmeans)}_kmeans.pkl"
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
        self.output_size = params.get('output_size')
        self.n_clusters = params.get('n_clusters')
        self.fit_kmeans = pk.load(
            open(os.path.join(filepath, params.get('kmeans_object')), 'rb'))
        self.initialized = True
