
from .base_decider import BaseDecider
from typing import Callable

import numpy as np
from enum import Enum

def all_agree(decisions: np.ndarray) -> bool:
    return np.all(decisions == decisions[0])

def at_least_n_agree(n: int) -> Callable[[np.ndarray], bool]:
    return lambda decisions: np.count_nonzero(decisions) >= n

class CombiningDecider(BaseDecider):

    def __init__(self, n, rule: Callable[[np.ndarray], bool]) -> None:
        super().__init__()
        self.n = n
        self.rule = rule

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1,-1,self.n]

    def check_input_dim(self, X) -> bool:
        return super().check_input_dim(X)
    

    def serialize(self):
        return super().serialize()
    
    def load(self) -> None:
        return super().load()