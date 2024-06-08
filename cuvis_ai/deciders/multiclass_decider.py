
from .base_decider import BaseDecider
from ..node import Node

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial, get_shape_without_batch

import numpy as np

class MultiClassDecider(BaseDecider):
    """Simple multi-class maximum decider.
    Given a matrix with N channels, chooses the channel with the highest value per spatial location.
    The result will be a single channel matrix with the indices of the chosen channels as values."""

    def __init__(self, n: int) -> None:
        """Create multi-class decider instance

        Parameters
        ----------
        n : int
            Number of classes
        """
        super().__init__()
        self.id = f"{self.__class__.__name__}-{str(uuid.uuid4())}"
        self.n = n

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply the maximum classification on the data.
        Parameters
        ----------
        X : np.ndarray
            Data to apply the classification on.
        Returns
        -------
        np.ndarray
            Classified data. Single channel matrix comprised of the channel indices of the chosen classes.
        """
        self._input_dim = get_shape_without_batch(X, ignore=[0,1])
        flatten_soft_output = flatten_batch_and_spatial(X)
        decisions = np.argmax(flatten_soft_output,axis=1)
        return unflatten_batch_and_spatial(decisions, X.shape)

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1,-1, self.n]

    def serialize(self):
        return super().serialize()
    
    def load(self) -> None:
        return super().load()
    

# TODO: How would this functionality be integrated into Deep Learning Methods and Models