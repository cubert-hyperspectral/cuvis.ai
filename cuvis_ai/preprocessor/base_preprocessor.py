from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import typing
import uuid

class Preprocessor(ABC):
    """
    Abstract class for data preprocessing.
    """

    @abstractmethod
    def fit(self, X):
        """
        Fit the preprocessor to the data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        self
        """
        pass