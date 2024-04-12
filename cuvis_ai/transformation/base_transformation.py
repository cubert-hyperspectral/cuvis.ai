from abc import ABC, abstractmethod, abstractproperty

class BaseTransformation(ABC):
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
    def check_output_dim(self, X):
        pass

    @abstractmethod
    def check_input_dim(self, X):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def load(self):
        pass