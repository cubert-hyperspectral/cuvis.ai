from .base_transformation import BaseTransformation
from ..node import MetadataConsumer, MetadataConsumerInference, Node
import numpy as np
import uuid
import yaml
from typing import Dict, Iterable, Any, Tuple, List
import torch


class Reflectance(BaseTransformation, MetadataConsumer, MetadataConsumerInference):
    """Generic reflectance calculus: (data - dark) / (white - dark)
    Requires "Dark" and "White" references to be set in Metadata.

    Parameters
    ----------
    lower_bound : float, optional
        Threshold for the smallest allowed value, everything lower will be clamped. Set to None to allow any value. Default: 0.0
    upper_bound : float, optional
        Threshold for the largest allowed value, everything higher will be clamped. Set to None to allow any value. Default: 2.0
    """

    def __init__(self, lower_bound: float = 0.0, upper_bound: float = 2.0):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input_size = None
        self.output_size = None

    def fit(self, X: Tuple):
        pass

    def forward(self, X: Tuple[np.ndarray, List[Dict]]):
        """Apply reflectance calculus to the data.
        Returns the data as percentage values between the "Dark" and "White" references set in the meta-data.
        e.g. A pixel value of 1.0 means that the pixel is as bright as the white reference at this pixel, 1.5 -> 50% brighter, 0.0 -> as bright as the dark reference, -0.2 -> 20% darker than the dark reference.
        The output values can be clamped by setting :attr:`lower_bond` and :attr:`upper_bound`.


        Parameters
        ----------
        X : Tuple
            Data to compute reflectance of. Expects a tuple of (data, meta-data) as (np.ndarray, Dict)

        Returns
        -------
        Tuple
            Returns the reflectance data in a tuple along with the remaining data passed in.
        """

        def reflectanceCalc(cube: np.ndarray, white: np.ndarray, dark: np.ndarray, ub: Optional[float], lb: Optional[float]) -> np.ndarray:
            ref = np.divide(np.subtract(cube, dark), np.subtract(white, dark))
            if not ((self.lower_bound is None) and (self.upper_bound is None)):
                ref = torch.clamp(torch.as_tensor(
                    ref), self.lower_bound, self.upper_bound).numpy()
            return ref

        if not isinstance(X, tuple) or (isinstance(X, tuple) and len(X) != 2):
            raise ValueError(
                "Reflectance calculation input must be a tuple containing cube data and metadata containing dark and white references.")

        cubes = np.split(X[0], axis=0)
        metas = X[1]
        refs = []

        for cube, meta in zip(cubes, metas):
            try:
                dark = meta["references"]["Dark"]
                white = meta["references"]["White"]
            except KeyError:
                pass
            except AttributeError:
                pass
            if dark is None or white is None:
                raise ValueError(
                    "Reflectance calculation requires a dark and white references in the metadata to be present.")
            refs.append(reflectanceCalc(cube, white, dark,
                        self.upper_bound, self.lower_bound))

        return np.stack(refs, axis=0)

    @Node.output_dim.getter
    def output_dim(self) -> Tuple[int, int, int]:
        return (-1, -1, -1)

    @Node.input_dim.getter
    def input_dim(self) -> Tuple[int, int, int]:
        return (-1, -1, -1)

    def serialize(self, serial_dir: str) -> str:
        """Serialize this node."""
        data = {
            "type": type(self).__name__,
            "lower": self.lower_bound,
            "upper": self.upper_bound,
        }
        return yaml.dump(data, default_flow_style=False)

    def load(self, filepath: str, params: Dict):
        """Load this node from a serialized graph."""
        try:
            self.lower_bound = float(params["lower"])
        except:
            self.lower_bound = None
        try:
            self.upper_bound = float(params["upper"])
        except:
            self.upper_bound = None
