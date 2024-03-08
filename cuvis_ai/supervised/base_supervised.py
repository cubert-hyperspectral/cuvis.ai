from abc import ABC, abstractmethod

import numpy as np
from typing import Dict

class BaseSupervised(ABC):

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def check_input_dim(self,X):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def load(self):
        pass
