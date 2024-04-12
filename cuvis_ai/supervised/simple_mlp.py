from .base_supervised import BaseSupervised
from sklearn.neural_network import MLPClassifier as sk_mlp
from ..utils.nn_config import Optimizer
from ..utils.numpy_utils import flatten_spatial, flatten_labels, unflatten_spatial
import numpy as np

from dataclasses import dataclass

@dataclass
class MLP(BaseSupervised):
    epochs: int = 10
    optimizer: Optimizer = None
    verbose: bool = False

    def __post_init__(self):
        self.initialized = False

        args = dict()
        if self.optimizer is not None:
            args.update(self.optimizer.args)

        self.mlp = sk_mlp(max_iter=self.epochs,verbose=self.verbose,**args)


    def fit(self, X: np.ndarray, Y: np.ndarray):
        flatten_image = flatten_spatial(X)

        flatten_l = flatten_labels(Y)

        print(f'shape image: {flatten_image.shape}')
        print(f'shape labels: {flatten_l.shape}')

        self.mlp.fit(flatten_image,flatten_l)

    def check_input_dim(self, X: np.ndarray):
        pass
    
    def forward(self, X: np.ndarray):
        flatten_image = flatten_spatial(X)

        predictions = self.mlp.predict(flatten_image)
        predictions = unflatten_spatial(predictions, X.shape)
        return predictions

    def serialize(self):
        pass

    def load():
        pass

