
from .base_decider import BaseDecider

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial

import numpy as np

class BinaryDecider(BaseDecider):

    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, X: np.ndarray) -> np.ndarray:

        flatten_soft_output = flatten_batch_and_spatial(X)
        decisions = flatten_soft_output >= self.threshold
        return unflatten_batch_and_spatial(decisions, X.shape)

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1,-1,1]

    def serialize(self):
        return super().serialize()
    
    def load(self) -> None:
        return super().load()
    

# TODO: How would this functionality be integrated into Deep Learning Methods and Models