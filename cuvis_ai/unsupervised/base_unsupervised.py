from abc import ABC, abstractmethod
import numpy as np
import uuid

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