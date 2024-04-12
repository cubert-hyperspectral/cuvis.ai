from abc import ABC, abstractmethod
import numpy as np

class BaseDecider(ABC):
    """
    Abstract class for Decision Making Nodes.

    The decider nodes transform a prediction state into a final prediction
    based on the task that needs to be accomplished.
    """
    
    @abstractmethod
    def forward(self, X):
        """
        Predict labels based on the input labels.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        pass


    @property
    @abstractmethod
    def input_dim(self) -> tuple[int,int,int]:
        """
        Returns the needed shape for the input data.
        If a dimension is not important it will return -1 in the specific position.

        Returns:
        (tuple) needed shape for data
        """
        pass


    @abstractmethod
    def check_input_dim(self, X) -> bool:
        """
        Check that the parameters for the input data data match user
        expectations

        Parameters:
        X (array-like): Input data.

        Returns:
        (Bool) Valid data 
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