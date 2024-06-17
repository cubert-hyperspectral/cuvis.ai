import os
import yaml
import uuid
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from ..node import Node
from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial
from typing import Union, Optional, Callable
from .base_unsupervised import BaseUnsupervised
from sklearn.cluster import MeanShift as sk_meanshift


class MeanShift(Node, BaseUnsupervised):
    """Mean Shift Clustering

    Parameters
    ----------
    Node : Abstract Node, shared by all CUVIS.AI classes
    BaseUnsupervised : Secondary inheritance for unsupervised nodes 
    """

    def __init__(self):
        """Initialize a Mean Shift clustering algorithm.
        """
        super().__init__()
        self.input_size = None
        self.initialized = False
        self.input_size = (-1, -1, -1)
        self.output_size = (-1, -1, -1)

    def fit(self, X: np.ndarray):
        """Train the Mean Shift classifier given a sample datacube.

        Parameters
        ----------
        X : np.ndarray
            Training data for classifier in shape of W x H x C
        """
        image_2d = flatten_batch_and_spatial(X)
        self.fit_meanshift = sk_meanshift()
        self.fit_meanshift.fit(image_2d)
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
        """Apply Mean Shift classifier to new data

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
        data = self.fit_meanshift.predict(image_2d)
        return unflatten_batch_and_spatial(data, X.shape)

    def serialize(self, serial_dir: str) -> str:
        """Write the model parameters to a YAML format and save Mean Shift weights

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
        pk.dump(self.fit_meanshift, open(os.path.join(
            serial_dir, f"{hash(self.fit_meanshift)}_mean_shift.pkl"), "wb"))
        data = {
            'type': type(self).__name__,
            'id': self.id,
            'input_size': self.input_size,
            'mean_shift_object': f"{hash(self.fit_meanshift)}_mean_shift.pkl"
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
        self.fit_meanshift = pk.load(
            open(os.path.join(filepath, params.get('mean_shift_object')), 'rb'))
        self.initialized = True
