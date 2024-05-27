from abc import ABC, abstractmethod
import numpy as np
import uuid
from ..utils.numpy_utils import get_shape_without_batch, check_array_shape

class BaseDecider(ABC):
    """
    Abstract class for Decision Making Nodes.

    The decider nodes transform a prediction state into a final prediction
    based on the task that needs to be accomplished.
    """
    
    def __init__(self):
        self.input_size = None
        self.output_size = None
        self.id = str(uuid.uuid4())
        
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


    @property
    @abstractmethod
    def input_dim(self) -> tuple[int,int,int]:
        """
        Returns the needed shape for the input data.
        If a dimension is not important it will return -1 in the specific position.

        Returns
        -------
        tuple
            Needed shape for data
        """
        pass


    def check_input_dim(self, X) -> bool:
        """
        Check that the parameters for the input data data match user
        expectations

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        Bool
            Valid data 
        """
        return check_array_shape(get_shape_without_batch(X), self.input_dim)

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