from abc import ABC, abstractmethod
from ..node import Node, CubeConsumer
from ..utils.numpy import flatten_spatial, unflatten_spatial
import numpy as np
import yaml
from copy import deepcopy


class AbstractDetector(Node, CubeConsumer, ABC):
    """
    Abstract class for data statistical based anomaly detectors.
    """

    def __init__(self, ref_spectra: list = []):
        super().__init__()
        self.ref_spectra = self.spectra_to_array(ref_spectra)
        self.initialized = False

    @staticmethod
    def spectra_to_array(ref_spectra: np.ndarray | list) -> np.ndarray:
        if isinstance(ref_spectra, list):
            ref_spectra = np.array(ref_spectra)
            if ref_spectra.ndim == 1:
                ref_spectra = ref_spectra.reshape((1, -1))
        if ref_spectra.ndim == 1:
            ref_spectra = ref_spectra[np.newaxis, :]
        return ref_spectra

    def fit(self, X: np.ndarray):
        self.initialized = True
        return self

    def forward(self, X: np.ndarray, ref_spectra: list = None) -> np.ndarray:
        ref = self.ref_spectra if ref_spectra is None else self.spectra_to_array(ref_spectra)
        if ref.size > 0 or self._allow_refless:
            if X.shape[-1] != (ref.shape[-1] if ref.size > 0 else X.shape[-1]):
                raise ValueError("Mismatch in input data and reference spectra!")
            # flatten spatial dims
            flat = flatten_spatial(X)
            scores = self.score(flat, ref)
            # reshape back to spatial cube
            return scores.reshape(*X.shape[:-1], scores.shape[-1])
        else:
            raise ValueError("No reference spectra provided and refless not enabled!")

    def serialize(self, working_dir: str) -> str:
        data = deepcopy(self.__dict__)
        data['type'] = type(self).__name__
        data['ref_spectra'] = data['ref_spectra'].tolist()
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str = None):
        params = params.copy()
        params.pop('type', None)
        self.__dict__.update(params)
        self.ref_spectra = np.array(self.ref_spectra)
        return self

    @abstractmethod
    def score(self, data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Compute anomaly score(s) for flat data or full cube."""
        pass

    @property
    def _allow_refless(self) -> bool:
        return False

    @Node.input_dim.getter
    def input_dim(self) -> list:
        if self.ref_spectra.size > 0:
            return [-1, -1, self.ref_spectra.shape[1]]
        else:
            return [-1, -1, -1]

    @Node.output_dim.getter
    def output_dim(self) -> list:
        return [-1, -1, 1]
