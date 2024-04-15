from abc import ABC, abstractmethod
import numpy as np

from ..utils.numpy_utils import check_array_shape, get_shape_without_batch

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
        pass

    @property
    @abstractmethod
    def output_dim(self) -> tuple[int,int,int]:
        pass

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