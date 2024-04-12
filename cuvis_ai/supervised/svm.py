from .base_supervised import BaseSupervised
from sklearn import svm as sk_svm

from ..utils.numpy_utils import flatten_batch_and_spatial, flatten_batch_and_labels, unflatten_batch_and_spatial, get_shape_without_batch

import numpy as np

class SVM(BaseSupervised):

    def __init__(self) -> None:
        super().__init__()

        self.svm = None

    @BaseSupervised.output_dim.getter
    def input_dim(self):
        return self._input_dim
    
    @BaseSupervised.output_dim.getter
    def input_dim(self):
        return self._output_dim

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._input_dim = get_shape_without_batch(X, ignore=[0,1])

        flatten_image = flatten_batch_and_spatial(X)
        flatten_l = flatten_batch_and_labels(Y)

        self.svm.fit(flatten_image,flatten_l)

        self.input_size = X.shape[2] 
        self.initialized = True

    def check_input_dim(self, X: np.ndarray):
        pass
    
    def forward(self, X: np.ndarray):
        flatten_image = flatten_batch_and_spatial(X)

        predictions = self.svm.predict(flatten_image)
        predictions = unflatten_batch_and_spatial(predictions, X.shape)
        return predictions

    def serialize(self):
        pass

    def load():
        pass

