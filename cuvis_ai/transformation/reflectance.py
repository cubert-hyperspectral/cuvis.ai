from ..node.base import BaseTransformation
from ..node import Node
import numpy as np


class Reflectance(Node, BaseTransformation):
    """Generic reflectance calculus: (data - dark) / (white - dark)
    Requires "Dark" and "White" references to be set in Metadata.

    Parameters
    ----------
    lower_bound : float, optional
        Threshold for the smallest allowed value, everything lower will be clamped. Set to None to allow any value. Default: 0.0
    upper_bound : float, optional
        Threshold for the largest allowed value, everything higher will be clamped. Set to None to allow any value. Default: 2.0
    """

    def __init__(self, white_ref: np.ndarray = None, dark_ref: np.ndarray = None, lower_bound: float = 0.0, upper_bound: float = 2.0):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.white_ref = white_ref
        self.dark_ref = dark_ref
        if not self.white_ref is None and not self.dark_ref is None:
            if self.white_ref.shape != self.dark_ref.shape:
                raise ValueError(
                    "Mismatching Dimensions in Dark and White Reference")

        self.set_forward_meta_request(
            references__Dark=(dark_ref is None), references__White=(white_ref is None))

    def forward(self, X: np.ndarray, references__White: np.ndarray = None, references__Dark: np.ndarray = None):
        """Apply reflectance calculus to the data.
        Returns the data as percentage values between the "Dark" and "White" references set in the meta-data.
        e.g. A pixel value of 1.0 means that the pixel is as bright as the white reference at this pixel, 1.5 -> 50% brighter, 0.0 -> as bright as the dark reference, -0.2 -> 20% darker than the dark reference.
        The output values can be clamped by setting :attr:`lower_bond` and :attr:`upper_bound`.

        Parameters
            ----------
            X : np.ndarray
                The input data array for which reflectance is computed. Must have the same shape 
                as `references__White` and `references__Dark`.
            references__White : np.ndarray
                The white reference array. Defines the maximum intensity for each pixel.
            references__Dark : np.ndarray
                The dark reference array. Defines the minimum intensity for each pixel.

            Returns
            -------
            np.ndarray
                An array of reflectance values with the same shape as the input `X`. The reflectance 
                values are computed as `(X - references__Dark) / (references__White - references__Dark)` 
                and optionally clamped between the lower and upper bounds.

        """

        if (self.dark_ref is None and references__Dark is None) or (self.white_ref is None and references__White is None):
            raise RuntimeError(
                "Missing References for Reflectance Normalization")

        dark_reference = references__Dark or np.resize(self.dark_ref, X.shape)
        white_reference = references__White or np.resize(
            self.white_ref, X.shape)

        numerator = X - dark_reference
        denominator = white_reference - dark_reference
        # Avoid division by zero
        reflectance = np.divide(numerator, denominator, where=denominator != 0)
        reflectance = np.nan_to_num(reflectance)

        # Clamp values if bounds are set
        if self.lower_bound is not None or self.upper_bound is not None:
            lower_bound = self.lower_bound if self.lower_bound is not None else -np.inf
            upper_bound = self.upper_bound if self.upper_bound is not None else np.inf
            reflectance = np.clip(reflectance, lower_bound, upper_bound)

        return reflectance

    @Node.output_dim.getter
    def output_dim(self) -> tuple[int, int, int]:
        return (-1, -1, -1)

    @Node.input_dim.getter
    def input_dim(self) -> tuple[int, int, int]:
        return (-1, -1, -1)

    def serialize(self, serial_dir: str) -> dict:
        """Serialize this node."""
        data = {
            "lower": self.lower_bound,
            "upper": self.upper_bound,
        }
        return data

    def load(self, params: dict, serial_dir: str) -> None:
        """Load this node from a serialized graph."""
        try:
            self.lower_bound = float(params["lower"])
        except:
            self.lower_bound = None
        try:
            self.upper_bound = float(params["upper"])
        except:
            self.upper_bound = None
