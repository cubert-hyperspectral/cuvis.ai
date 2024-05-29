
from .base_decider import BaseDecider
from typing import Callable

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial

import numpy as np

def all_agree(decisions: np.ndarray) -> bool:
    return np.all(decisions == decisions[0])

def at_least_n_agree(n: int) -> Callable[[np.ndarray], bool]:
    return lambda decisions: np.count_nonzero(decisions) >= n

class CombiningDecider(BaseDecider):
    """Decider using values of multiple channels to classify the result.
    The data of all channels at a spatial location are utilized in the chosen decision strategy to classify each data point.
    
    Parameters
    ----------
    channel_count : int
        The number of channels to expect
    rule : Callable[[np.ndarray], bool]
        The decision strategy to use. :meth:`all_agree` and :meth:`at_least_n_agree` are provided here.
        Custom strategies may also be used.
    """

    def __init__(self, channel_count:int, rule: Callable[[np.ndarray], bool]) -> None:
        super().__init__()
        self.id = F"{self.__class__.__name__}-{str(uuid.uuid4())}"
        self.n = channel_count
        self.rule = np.vectorize(rule)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply the chosen :arg:`rule` to the input data.
        Parameters
        ----------
        X : np.ndarray
            Data to classify.
        Returns
        -------
        np.ndarray :
            Data classified to a single channel boolean matrix.
        """
        flatten_soft_output = flatten_batch_and_spatial(X)
        decisions = self.rule(flatten_soft_output)
        return unflatten_batch_and_spatial(decisions, X.shape)

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1,-1,self.n]   

    def serialize(self):
        return super().serialize()
    
    def load(self) -> None:
        return super().load()