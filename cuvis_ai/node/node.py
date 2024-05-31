from abc import ABC, abstractmethod
import numpy as np
import typing
import uuid

from ..utils.numpy_utils import get_shape_without_batch, check_array_shape

class Node(ABC):
    """
    Abstract class for data preprocessing.
    """
    def __init__(self):
        self.id =  f'{type(self).__name__}-{str(uuid.uuid4())}'

    @abstractmethod
    def forward(self, X):
        """Forwards the data through the Node

        Parameters
        ----------
        X : array-like
            Input data
        """
        pass

    def check_output_dim(self, X) -> bool:
        """Check that the parameters for the output data data match user
        expectations.

        Parameters
        ----------
        X : array-like
            Output data

        Returns
        -------
        bool
            Returns if the data fits the expected dimensionality
        """
        return check_array_shape(get_shape_without_batch(X), self.output_dim)

    def check_input_dim(self, X):
        """Check that the parameters for the input data data match user
        expectations.

        Parameters
        ----------
        X : array-like
            Input data

        Returns
        -------
        bool
            Returns if the data fits the expected dimensionality
        """
        return check_array_shape(get_shape_without_batch(X), self.input_dim)

    @property
    @abstractmethod
    def input_dim(self) -> tuple[int,int,int]:
        """Returns the needed shape for the input data.
        If a dimension is not important, it will return -1 in the specific position.

        Returns
        -------
        tuple[int,int,int]
            The shape of the data that is expected
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> tuple[int,int,int]:
        """Returns the shape for the output data.
        If a dimension is dependent on the input, it will return -1 in the specific position.

        Returns
        -------
        tuple[int,int,int]
            The shape of the data that is expected
        """
        pass

    @abstractmethod
    def serialize(self):
        """Convert the class into a serialized representation
        """
        pass

    @abstractmethod
    def load(self):
        """Load from serialized format into an object
        """
        pass