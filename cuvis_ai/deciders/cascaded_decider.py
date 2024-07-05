from .base_decider import BaseDecider

import numpy as np


class Cascaded(BaseDecider):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1, -1, 1]

    @BaseDecider.output_dim.getter
    def output_dim(self):
        """
        Returns the provided shape for the output data.
        If a dimension is not important it will return -1 in the specific position.

        Returns
        -------
        tuple
            Provided shape for data
        """
        return [-1, -1, 1]

    def serialize(self):
        return super().serialize()

    def load(self) -> None:
        return super().load()
