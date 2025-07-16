from .AbstractDetector import AbstractDetector
import numpy as np
import warnings
from ..utils.numpy import flatten_spatial, unflatten_spatial

class RXDetector(AbstractDetector):
    """
    Reed-Xiaoli (RX) anomaly detector using Mahalanobis distance.

    Works on either a 3D data cube (H x W x B) or flattened pixel array (N x B).
    If no reference spectra are provided, the detector estimates background
    statistics from the input data itself.
    """

    def __init__(self, ref_spectra: list = None):
        super().__init__(ref_spectra or [])

    @property
    def _allow_refless(self) -> bool:
        return True

    def score(self, data: np.ndarray, ref_spectra: list = []) -> np.ndarray:
        # Handle full data cube input
        if data.ndim == 3:
            # data shape: (H, W, B)
            H, W, B = data.shape
            flat = flatten_spatial(data)
            flat_scores = self.score(flat, ref_spectra)
            return unflatten_spatial(flat_scores, (H,W,1))

        # Now 'data' is flattened: (N, B)
        # Estimate background mean/covariance
        if len(ref_spectra) == 0:
            mu = np.mean(data, axis=0)
            cov = np.cov(data, rowvar=False)
        else:
            mu = np.mean(ref_spectra, axis=0)
            cov = np.cov(ref_spectra, rowvar=False)

        # Warn if unnormalized data
        if np.percentile(data, 90) > 2.0:
            warnings.warn(
                "RX detector is being used without properly normalized data. "
                "Unexpected behavior may occur!"
            )

        # Invert covariance
        cov_inv = np.linalg.pinv(cov)
        diff = data - mu
        # Mahalanobis distance: diff * cov_inv * diff^T per sample
        m_dist = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
        # Return shape (N, 1)
        return m_dist[:, np.newaxis]

    def fit(self, X: np.ndarray):
        super().fit(X)
        return self
