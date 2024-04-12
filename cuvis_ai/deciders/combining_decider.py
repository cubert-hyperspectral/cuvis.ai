
from .base_decider import BaseDecider
from typing import Callable

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial

import numpy as np

def all_agree(decisions: np.ndarray) -> bool:
    return np.all(decisions == decisions[0])

def at_least_n_agree(n: int) -> Callable[[np.ndarray], bool]:
    return lambda decisions: np.count_nonzero(decisions) >= n

class CombiningDecider(BaseDecider):

    def __init__(self, n, rule: Callable[[np.ndarray], bool]) -> None:
        super().__init__()
        self.n = n
        self.rule = np.vectorize(rule)

    def forward(self, X: np.ndarray) -> np.ndarray:
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