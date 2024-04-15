from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import typing
import uuid

class Preprocessor(ABC):
    """
    Abstract class for data preprocessing.
    """
    def __init__(self):
        self.input_size = None
        self.output_size = None
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