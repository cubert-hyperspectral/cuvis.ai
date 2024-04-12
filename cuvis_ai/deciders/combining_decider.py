
from .base_decider import BaseDecider

import numpy as np

class CombiningDecider(BaseDecider):

    def __init__(self, n) -> None:
        super().__init__()
        self.n = n

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