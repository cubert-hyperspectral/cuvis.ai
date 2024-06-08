from abc import ABC, abstractmethod

import numpy as np
from ..utils.numpy_utils import check_array_shape, get_shape_without_batch
from ..node import CubeConsumer, LabelConsumer

class BaseSupervised(ABC, CubeConsumer, LabelConsumer):

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @property
    @abstractmethod
    def input_dim(self) -> tuple[int,int,int]:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> tuple[int,int,int]:
        pass

    def check_input_dim(self,X):
        return check_array_shape(get_shape_without_batch(X), self.input_dim)

    def check_output_dim(self, X):
        return check_array_shape(get_shape_without_batch(X), self.output_dim)

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def load(self):
        pass
