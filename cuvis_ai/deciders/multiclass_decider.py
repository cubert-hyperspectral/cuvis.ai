
from .base_decider import BaseDecider

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial, get_shape_without_batch

import numpy as np

class MultiClassDecider(BaseDecider):

    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._input_dim = get_shape_without_batch(X, ignore=[0,1])
        flatten_soft_output = flatten_batch_and_spatial(X)
        decisions = np.argmax(flatten_soft_output,axis=1)
        return unflatten_batch_and_spatial(decisions, X.shape)

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return self.input_dim

    def serialize(self):
        return super().serialize()
    
    def load(self) -> None:
        return super().load()
    

# TODO: How would this functionality be integrated into Deep Learning Methods and Models