from .base_supervised import BaseSupervised
from sklearn import svm as sk_svm

import numpy as np

class SVM(BaseSupervised):

    def __init__(self) -> None:
        super().__init__()

        self.svm = None




    def fit(self, X: np.ndarray, Y: np.ndarray):
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)

        self.svm.fit(image_2d,Y)

        self.input_size = X.shape[2] 
        self.initialized = True


    
    def check_input_dim(self, X: np.ndarray):
        pass
    
    def predict(self, X: np.ndarray):
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        data = self.svm.predict(image_2d)
        cube_data = data.reshape((X.shape[0], X.shape[1]))
        return cube_data

    def serialize(self):
        pass

    def load():
        pass

