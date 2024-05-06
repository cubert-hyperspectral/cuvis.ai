from abc import ABC, abstractmethod
from ..node import Node
from ..utils.numpy_utils import flatten_spatial, flatten_labels, unflatten_spatial
import numpy as np
import typing
from typing import List, Union
import uuid
import yaml
import warnings
from copy import deepcopy

class AbstractDistance(Node):
    """
    Abstract class for data preprocessing.

    There are two ways spectral distance can be used

    1) Distance relative to a reference spectra, which returns a single scored image
    2) Distance relative to multiple reference spectral, which returns an array of scored images
    (This can then be passed into a decider algorithm)
    # TODO: Should this behavior be allowed? This might break the notion of input/output dimensions
    - If the spectra are known, they can be stored with the object initially
    """
    def __init__(self, ref_spectra: List=[]):
        self.id =  f'{type(self).__name__}-{str(uuid.uuid4())}'
        # Assign any reference spectra, if they exist
        self.ref_spectra = self.spectra_to_array(ref_spectra)


    @staticmethod
    def spectra_to_array(ref_spectra: Union[np.ndarray, List]) -> np.ndarray:
        if type(ref_spectra) == list:
            # Cast to a numpy
            ref_spectra = np.array(ref_spectra)
        if ref_spectra.shape == 1:
            # Squeeze an extra dimension
            ref_spectra = np.expand_dims(ref_spectra, axis=0)
        return ref_spectra
    
    def forward(self, X, ref_spectra=[]):
        """
        Pass the data through comparative function

        Parameters:
        X (array-like): Input data.
        ref_spectra (array-like): Spectra to compare against
        Returns:
        self
        """
        # Default behavior is to use the ref_spectra passed to the function
        # This overrides any previously stored spectra
        if len(ref_spectra) > 0:
            # Process each spectra
            ref_spectra = self.spectra_to_array(ref_spectra)
            if X.shape[-1] != ref_spectra.shape[-1]:
                raise ValueError('Mismatch in input data and reference spectra!')
            # Process and return
            res = self.score(X.reshape(X.shape[0]*X.shape[1], X.shape[2]), ref_spectra)
            return res.reshape(X.shape[0], X.shape[1], len(ref_spectra))
        elif len(self.ref_spectra) > 0:
            if X.shape[-1] != self.ref_spectra.shape[-1]:
                raise ValueError('Mismatch in input data and reference spectra!')
            # Process and return
            res = self.score(flatten_spatial(X), self.ref_spectra)
            return res.reshape(X.shape[0], X.shape[1], len(self.ref_spectra))    
        else:
            raise ValueError('No reference spectra provided!')

    def serialize(self, working_dir: str):
        data = deepcopy(self.__dict__)
        data['type'] = type(self).__name__
        data['ref_spectra'] = data['ref_spectra'].tolist()
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str=None):
        '''
        Load dumped parameters to recreate the distance object
        '''
        # Delete the type param
        del params['type']
        self.__dict__ = params
        # Cast reference spectra back to numpy type
        self.ref_spectra = np.array(self.ref_spectra)

    @abstractmethod
    def score(self, data, X):
        pass

    @Node.input_dim.getter
    def input_dim(self):
        # Note: This denotes by default we don't care about the input image shape
        # so long as the wavelengths match
        if len(self.ref_spectra) > 0:
            return [-1,-1, self.ref_spectra.shape[1]]
        else:
            return [-1,-1,-1]
    
    @Node.output_dim.getter
    def output_dim(self):
        # Note: Output can be arbitrary length depending on the reference spectra
        return [-1,-1,-1]

class SpectralAngle(AbstractDistance):
    '''
    Cosine distance between spectra according to the Spectral Angle Formula

    Nota Bene: Measurements should be normalized as large values skews this calculation towards π/2
    '''
    def __init__(self, ref_spectra: List=[]):
        super().__init__(ref_spectra)

    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        # Throw a warning for a large number of unnormalized values
        if np.percentile(data, 90) > 2.0:
            # 10% of the data exceeds 200%
            warnings.warn("Spectral angle mapper is being used without properly normalized data. Unexpected behavior may occur!")
        output_scores = np.zeros((data.shape[0], ref_spectra.shape[0]))
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores[:, idx] = np.arccos(
                np.dot(data, ref_spectra[idx]) / (np.linalg.norm(data, axis=1) * np.linalg.norm(ref_spectra[idx]))
            )
        return output_scores

class Euclidean(AbstractDistance):
    '''
    L2 Distance
    '''
    def __init__(self, ref_spectra: List=[]):
        super().__init__(ref_spectra)
    
    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        output_scores = np.zeros((data.shape[0], ref_spectra.shape[0]))
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores[:, idx] = np.sqrt(np.sum((data - ref_spectra[idx,:])**2, axis=1))
        return output_scores

class Manhattan(AbstractDistance):
    '''
    L1 Distance
    '''
    def __init__(self, ref_spectra: List=[]):
        super().__init__(ref_spectra)

    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        output_scores = np.zeros((data.shape[0], ref_spectra.shape[0]))
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores[:, idx] =  np.sum(np.abs(data - ref_spectra[idx,:]), axis=1)
        return output_scores

class Canberra(AbstractDistance):
    '''
    Weighted L1 Distance
    '''
    def __init__(self, ref_spectra: List=[]):
        super().__init__(ref_spectra)

    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        output_scores = np.zeros((data.shape[0], ref_spectra.shape[0]))
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores[:, idx] =  np.sum(
                np.abs(data - ref_spectra[idx,:])/(np.abs(data) + np.abs(ref_spectra[idx,:])),
                axis=1)
        return output_scores

class Minkowski(AbstractDistance):
    '''
    Weighted L1 Distance
    '''
    def __init__(self,  degree: int, ref_spectra: List=[]):
        super().__init__(ref_spectra)
        self.degree = degree

    def score(self, data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        output_scores = np.zeros((data.shape[0], ref_spectra.shape[0]))
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores[:, idx] = (np.sum((data - ref_spectra[idx,:])**self.degree, axis=1))**(1.0/float(self.degree))
        return output_scores

class GFC(AbstractDistance):
    '''
    Goodness-of-fit Coefficient (GFC)
    '''
    def __init__(self, ref_spectra: List=[]):
        super().__init__(ref_spectra)

    def score(self, data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        output_scores = np.zeros((data.shape[0], ref_spectra.shape[0]))
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores[:, idx] = 1 - (
                np.dot(data, ref_spectra[idx]) / (np.linalg.norm(data, axis=1) * np.linalg.norm(ref_spectra[idx]))
            )
        return output_scores

class ECS(AbstractDistance):
    '''
    Euclidean Distance of Cumulative Spectrum (ECS)
    '''
    def __init__(self,  wavelengths: Union[np.ndarray, List], ref_spectra: List=[]):
        super().__init__(ref_spectra)
        # Cast this to a list, necessary for serialization
        self.wavelengths = list(wavelengths)

    @Node.input_dim.getter
    def input_dim(self):
        # Note: this function depends on the wavelengths, so we need to match the dimension
        return [-1,-1, len(self.wavelengths)]

    @Node.output_dim.getter
    def output_dim(self):
        # Note: Output can be arbitrary length depending on the reference spectra
        return [-1,-1,-1]

    def score(self, data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        output_scores = np.zeros((data.shape[0], ref_spectra.shape[0]))
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores[:, idx] = np.sqrt((np.trapz(data, self.wavelengths, axis=1) - np.trapz(ref_spectra[idx], self.wavelengths))**2)
        return output_scores
