from abc import ABC, abstractmethod
import uuid
import numpy as np
import uuid
from typing import Tuple
from ..utils.numpy_utils import get_shape_without_batch, check_array_shape
from ..node import Node
from ..node.Consumers import *


class BaseDecider(Node, CubeConsumer, ABC):
    """
    Abstract class for Decision Making Nodes.

    The decider nodes transform a prediction state into a final prediction
    based on the task that needs to be accomplished.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, X):
        """
        Predict labels based on the input labels.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        Any
            Transformed data.
        """
        pass

    @abstractmethod
    def serialize(self):
        """
        Convert the class into a serialized representation
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load from serialized format into an object
        """
        pass
