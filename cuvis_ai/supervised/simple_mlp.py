from .base_supervised import BaseSupervised
from sklearn.neural_network import MLPClassifier as sk_mlp
from ..utils.nn_config import Optimizer

import numpy as np

from dataclasses import dataclass

@dataclass
class MLP(BaseSupervised):
    epochs: int = 10
    optimizer: Optimizer = None



    def __post_init__(self):
        self.initialized = False

        args = dict()
        if self.optimizer is not None:
            args.update(self.optimizer.args)

        self.mlp = sk_mlp(max_iter=self.epochs,**args)


    def fit(self, X: np.ndarray, Y: np.ndarray):
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)

        labels_2d = Y.reshape(n_pixels)

        self.mlp.fit(image_2d,labels_2d)

        self.input_size = X.shape[2] 
        self.initialized = True


    
    def check_input_dim(self, X: np.ndarray):
        pass
    
    def predict(self, X: np.ndarray):
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        data = self.mlp.predict(image_2d)
        cube_data = data.reshape((X.shape[0], X.shape[1]))
        return cube_data

    def serialize(self):
        pass

    def load():
        pass

