from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import typing
from typing import Dict
import uuid

class BaseUnsupervised(ABC):
    """
    Abstract class for data preprocessing.
    """
    def __init__(self):
        self.id =  str(uuid.uuid4())

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