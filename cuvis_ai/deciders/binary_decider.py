
from .base_decider import BaseDecider

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial

import numpy as np
from typing import Any


class BinaryDecider(BaseDecider):
    """Simple decider node using a static threshold to classify data.

    Parameters
    ----------
    threshold : Any
        The threshold to use for classification: result = (input >= threshold)
    """

    def __init__(self, threshold: Any) -> None:
        super().__init__()
        self.threshold = threshold
        self.id = F"{self.__class__.__name__}-{str(uuid.uuid4())}"

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply binary decision on input data.

        Paramaters
        ----------
        X : np.ndarray
            Input data as numpy array

        Returns
        -------
        np.ndarray
            Classified input data. Where the datapoints are False if smaller than the threshold or True if larger or equal.
        """
        flatten_soft_output = flatten_batch_and_spatial(X)
        decisions = flatten_soft_output >= self.threshold
        return unflatten_batch_and_spatial(decisions, X.shape)

    @BaseDecider.input_dim.getter
    def input_dim(self):
        """
        Returns the needed shape for the input data.
        If a dimension is not important it will return -1 in the specific position.

        Returns
        -------
        tuple
            Needed shape for data
        """
        return [-1, -1, 1]

    def serialize(self):
        """
        Convert the class into a serialized representation
        """
        return super().serialize()

    def load(self) -> None:
        """
        Load from serialized format into an object
        """
        return super().load()


# TODO: How would this functionality be integrated into Deep Learning Methods and Models
