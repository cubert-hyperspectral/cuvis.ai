from .base_supervised import BaseSupervised
from skorch import NeuralNetClassifier
from ..utils.nn_config import Optimizer
from ..utils.numpy_utils import flatten_arrays, flatten_labels
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer as torch_optim

from dataclasses import dataclass

def unflatten_arrays(data: np.ndarray, orig_shape):
    return data.reshape(orig_shape)

@dataclass
class SkorchWrapper(BaseSupervised):
    epochs: int = 10
    optimizer: Optimizer | torch_optim = None
    verbose: bool = False
    model: nn.Module



    def __post_init__(self):
        self.initialized = False

        args = dict()
        if self.optimizer is not None:
            args.update(self.optimizer.args)

        self.classifier = NeuralNetClassifier(self.model,**args)


    def fit(self, X: np.ndarray, Y: np.ndarray):
        flatten_image, _ = flatten_arrays(X)

        flatten_l, _ = flatten_labels(Y)

        print(f'shape image: {flatten_image.shape}')
        print(f'shape labels: {flatten_l.shape}')

        self.classifier.fit(flatten_image,flatten_l)

    def check_input_dim(self, X: np.ndarray):
        pass
    
    def forward(self, X: np.ndarray):
        flatten_image, _ = flatten_arrays(X)

        flatten_l, label_shape = flatten_l(Y)

        predictions = self.classifier.predict(flatten_image,flatten_l)
        predictions = unflatten_arrays(predictions,label_shape)
        return predictions

    def serialize(self):
        pass

    def load():
        pass

