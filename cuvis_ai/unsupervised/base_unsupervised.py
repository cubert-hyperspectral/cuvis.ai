from abc import ABC, abstractmethod
import numpy as np
import typing
from typing import Dict

class BaseUnsupervised(ABC):
    """
    Abstract class for data preprocessing.
    """
    
    @abstractmethod
    def fit(self, X):
        """
        Fit the preprocessor to the data.

        Parameters:
        X (array-like): Input data.

        Returns:
        self
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict labels based on the input labels.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        pass

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

    @abstractmethod
    def visualize(self):
        """
        Visualize classified data
        """
        pass