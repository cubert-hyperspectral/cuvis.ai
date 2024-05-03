from .base_transformation import BaseTransformation
from ..data import Metadata
import torch
from typing import Dict, Iterable

class Reflectance(BaseTransformation):
    """Generic reflectance calculus: (data - dark) / (white - dark)
    Requires "Dark" and "White" references to be set in Metadata.
    
    Args:
        lower_bound: Threshold for the smallest allowed value, everything lower will be clamped. Set to None to allow any value. Default: 0.0
        upper_bound: Threshold for the largest allowed value, everything higher will be clamped. Set to None to allow any value. Default: 2.0
    """
    
    def __init__(self, lower_bound:float=0.0, upper_bound:float=2.0):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.input_size = None
        self.output_size = None

    def fit(self, X):
        pass
    
    def forward(self, X):
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

    def check_output_dim(self, X):
        pass

    def check_input_dim(self, X):
        pass

    def serialize(self, serial_dir: str):
        data = {
            "type": type(self).__name__,
            "lower": self.lower_bound,
            "upper": self.upper_bound,
        }
        return yaml.dump(data, default_flow_style=False)

    def load(self, filepath:str, params:Dict):
        try:
            self.lower_bound = float(params["lower"])
        except:
            self.lower_bound = None
        try:
            self.upper_bound = float(params["upper"])
        except:
            self.upper_bound = None
        