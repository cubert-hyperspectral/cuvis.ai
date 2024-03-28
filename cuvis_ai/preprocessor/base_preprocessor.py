from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import typing


class Preprocessor(ABC):
    """
    Abstract class for data preprocessing.
    """
    def __init__(self):
        self.input_size = None
        self.output_size = None

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