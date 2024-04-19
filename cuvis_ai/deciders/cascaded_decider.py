from .base_decider import BaseDecider

import numpy as np

class Cascaded(BaseDecider):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1,-1,1]

    def serialize(self):
        return super().serialize()
    
    def load(self) -> None:
        return super().load()