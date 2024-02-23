import os
import yaml
import pickle as pk
from .base_preprocessor import Preprocessor
from sklearn.decomposition import PCA as sk_pca

class PCA(Preprocessor):
    """
    Principal Component Analysis (PCA) preprocessor.
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.input_size = None
        self.output_size = None
        self.initialized = False
        
    def fit(self, X):
        """
        Fit PCA to the data.

        Parameters:
        X (array-like): Input data.

        Returns:
        self
        """
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        self.fit_pca = sk_pca(n_components=self.n_components)
        self.fit_pca.fit(image_2d)
        # Set the dimensions for a later check
        self.input_size = X.shape[2] # Constrain the number of wavelengths
        self.output_size = self.n_components
        # Initialization is complete
        self.initialized = True

    def check_input_dim(self, X):
        assert(X.shape[2] == self.input_size)

    def check_output_dim(self, X):
        assert(X.shape[2] == self.n_components)
    
    def transform(self, X):
        """
        Transform the input data.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        # Transform data using precomputed PCA components
        n_pixels = X.shape[0] * X.shape[1]
        image_2d = X.reshape(n_pixels, -1)
        data = self.fit_pca.transform(image_2d)
        cube_data = data.reshape((X.shape[0], X.shape[1], self.n_components))
        return cube_data

    def serialize(self, serial_dir):
        '''
        This method should dump parameters to a yaml file format
        '''
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        # Write pickle object to file
        pk.dump(self.fit_pca, open(os.path.join(serial_dir,"pca.pkl"),"wb"))
        data = {
            'n_components': self.n_components,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'pca_object': os.path.join(serial_dir,"pca.pkl")
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, parameters):
        '''
        Load dumped parameters to recreate the pca object
        '''
        params = yaml.safe_load(parameters)
        self.input_size = params.get('input_size')
        self.n_components = params.get('n_components')
        self.output_size = params.get('output_size')
        self.fit_pca = pk.load(open(params.get('pca_object'),'rb'))
        self.initialized = True

