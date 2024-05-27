from .base_supervised import BaseSupervised
from sklearn.neural_network import MLPClassifier as sk_mlp
from ..utils.nn_config import Optimizer
from ..utils.numpy_utils import flatten_batch_and_spatial, flatten_batch_and_labels, unflatten_batch_and_spatial, get_shape_without_batch
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

    @BaseSupervised.input_dim.getter
    def input_dim(self):
        return self._input_dim
    
    @BaseSupervised.output_dim.getter
    def output_dim(self):
        return self._output_dim

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._input_dim = get_shape_without_batch(X, ignore=[0,1])

        flatten_image = flatten_batch_and_spatial(X)

        flatten_l = flatten_batch_and_labels(Y)

        print(f'shape image: {flatten_image.shape}')
        print(f'shape labels: {flatten_l.shape}')

        self.mlp.fit(flatten_image,flatten_l)
    
    def forward(self, X: np.ndarray):
        flatten_image = flatten_batch_and_spatial(X)

        predictions = self.mlp.predict(flatten_image)
        predictions = unflatten_batch_and_spatial(predictions, X.shape)
        return predictions

    def serialize(self):
        pass

    def load():
        pass

