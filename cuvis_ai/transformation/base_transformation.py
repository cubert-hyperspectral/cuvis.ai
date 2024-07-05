from abc import abstractmethod
from typing import Tuple
from ..node import CubeConsumer, Node
import uuid


class BaseTransformation(CubeConsumer):
    def __init__(self):
        self.input_size = None
        self.output_size = None

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def output_dim(self) -> Tuple[int, int, int]:
        pass

    @abstractmethod
    def input_dim(self) -> Tuple[int, int, int]:
        pass

    @abstractmethod
    def input_dim(self):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def load(self):
        pass
