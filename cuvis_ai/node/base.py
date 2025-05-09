from abc import ABC, abstractmethod
import numpy as np
from typing import Any

from ..node import CubeConsumer, LabelConsumer


class Preprocessor(ABC, CubeConsumer):
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

    @abstractmethod
    def forward(self, X):
        pass


class BaseSupervised(ABC, CubeConsumer, LabelConsumer):

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def forward(self, X):
        pass


class BaseUnsupervised(ABC, CubeConsumer):
    """Abstract node for all unsupervised classifiers to follow.

    Parameters
    ----------
    ABC : ABC
        Defines node as a base class.
    """

    @abstractmethod
    def fit(self, X: Any):
        """_summary_

        Parameters
        ----------
        X : Any
            Generic method to initialize a classifier with data.
        """
        pass

    @abstractmethod
    def forward(self, Any) -> Any:
        """Transform 

        Parameters
        ----------
        X : Any
            Generic method to pass new data through the unsupervised classifier.

        Returns
        -------
        Any
            Return type and shape must be defined by the implemented child classes.
        """
        pass


class BaseTransformation(CubeConsumer):

    @abstractmethod
    def forward(self, X):
        pass
