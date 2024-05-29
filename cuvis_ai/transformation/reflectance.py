from .base_transformation import BaseTransformation
from ..data import Metadata
import torch
from typing import Dict, Iterable, Any, Tuple

class Reflectance(BaseTransformation):
    """Generic reflectance calculus: (data - dark) / (white - dark)
    Requires "Dark" and "White" references to be set in Metadata.
    
    Parameters
    ----------
    lower_bound : float, optional
        Threshold for the smallest allowed value, everything lower will be clamped. Set to None to allow any value. Default: 0.0
    upper_bound : float, optional
        Threshold for the largest allowed value, everything higher will be clamped. Set to None to allow any value. Default: 2.0
    """
    
    def __init__(self, lower_bound:float=0.0, upper_bound:float=2.0):
        self.id = F"{self.__class__.__name__}-{str(uuid.uuid4())}"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input_size = None
        self.output_size = None

    def fit(self, X:Iterable):
        pass
    
    def forward(self, X:Iterable):
        """Apply reflectance calculus to the data.
        Returns the data as percentage values between the "Dark" and "White" references set in the meta-data.
        e.g. A pixel value of 1.0 means that the pixel is as bright as the white reference at this pixel, 1.5 -> 50% brighter, 0.0 -> as bright as the dark reference, -0.2 -> 20% darker than the dark reference.
        The output values can be clamped by setting :attr:`lower_bond` and :attr:`upper_bound`.
        
        
        Parameters
        ----------
        X : Iterable
            Data to compute reflectance of. Expects a tuple/list of (data, meta-data dict, ...) as (torch.Tensor, Dict, ...)
        
        Returns
        -------
        Tuple
            Returns the reflectance data in a tuple along with the remaining data passed in.
        """
        dark = None
        white = None
        if isinstance(X, Iterable):
            cube = X[0]
            metadata:Metadata = X[1]
            
            try:
                dark = metadata.references["Dark"]
                white = metadata.references["White"]
            except KeyError:
                pass
        if dark is None or white is None:
            raise ValueError("Reflectance calculation requires a dark and white references in the metadata to be present.")
        
        ref = torch.divide(torch.subtract(cube, dark), torch.subtract(white, dark))
        
        if not ((self.lower_bound is None) and (self.upper_bound is None)):
            ref = torch.clamp(ref, self.lower_bound, self.upper_bound)
        
        return (ref, *X[1:])

    def check_output_dim(self, X:Iterable):
        pass

    def check_input_dim(self, X:Iterable):
        pass

    def serialize(self, serial_dir:str):
        """Serialize this node."""
        data = {
            "type": type(self).__name__,
            "lower": self.lower_bound,
            "upper": self.upper_bound,
        }
        return yaml.dump(data, default_flow_style=False)

    def load(self, filepath:str, params:Dict):
        """Load this node from a serialized graph."""
        try:
            self.lower_bound = float(params["lower"])
        except:
            self.lower_bound = None
        try:
            self.upper_bound = float(params["upper"])
        except:
            self.upper_bound = None
        