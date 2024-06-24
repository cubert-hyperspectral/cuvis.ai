from abc import ABC, abstractmethod
from ..utils.numpy_utils import check_array_shape, get_shape_without_batch
from ..node import CubeConsumer, LabelConsumer


class BaseSupervised(ABC, CubeConsumer, LabelConsumer):

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def forward(self, X):
        pass
