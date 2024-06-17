from abc import ABC, abstractmethod
import numpy as np
import uuid
from typing import Any

from ..utils.numpy_utils import check_array_shape, get_shape_without_batch
from ..node import CubeConsumer

class BaseUnsupervised(ABC, CubeConsumer):
    """Abstract node for all unsupervised classifiers to follow.

    Parameters
    ----------
    ABC : ABC
        Defines node as a base class.
    """
    def __init__(self):
        """Initialize node
        """
        self.id =  str(uuid.uuid4())

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