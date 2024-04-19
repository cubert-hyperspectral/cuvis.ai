from abc import ABC, abstractmethod
import numpy as np
import typing

from ..utils.numpy_utils import get_shape_without_batch, check_array_shape

class Node(ABC):
    """
    Abstract class for data preprocessing.
    """
    def __init__(self):
        self.id = None

    @abstractmethod
    def forward(self, X):
        """
        Transform the input data.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        pass

    @abstractmethod
    def check_output_dim(self, X):
        """
        Check that the parameters for the output data data match user
        expectations

        Parameters:
        X (array-like): Input data.

        Returns:
        (Bool) Valid data 
        """
        return check_array_shape(get_shape_without_batch(X), self.output_dim)

    @abstractmethod
    def check_input_dim(self, X):
        """
        Check that the parameters for the input data data match user
        expectations

        Parameters:
        X (array-like): Input data.

        Returns:
        (Bool) Valid data 
        """
        return check_array_shape(get_shape_without_batch(X), self.input_dim)

    @property
    @abstractmethod
    def input_dim(self) -> tuple[int,int,int]:
        """
        Returns the needed shape for the input data.
        If a dimension is not important, it will return -1 in the specific position.

        Returns:
        (tuple) needed shape for data
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> tuple[int,int,int]:
        """
        Returns the shape for the output data.
        If a dimension is dependent on the input, it will return -1 in the specific position.

        Returns:
        (tuple) expected output shape for data
        """
        pass

    @abstractmethod
    def serialize(self):
        """
        Convert the class into a serialized representation
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load from serialized format into an object
        """
        pass