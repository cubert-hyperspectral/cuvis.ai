from .base_transformation import BaseTransformation
import torch
from typing import Dict

class Reflectance(BaseTransformation):
    def __init__(self):
        self.input_size = None
        self.output_size = None

    def fit(self, X):
        pass
    
    def forward(self, X):
        dark = None
        white = None
        if isinstance(X, tuple):
            c, m, l = X
            try:
                dark = l["references"]["Dark"]
                white = l["references"]["White"]
            except KeyError:
                pass
        if dark is None or white is None:
            raise ValueError("Reflectance calculation requires a dark and white references in the label data to be present.")
        
        c = torch.divide(torch.subtract(c, dark), torch.subtract(white, dark))
        return (c, m, l)

    def check_output_dim(self, X):
        pass

    def check_input_dim(self, X):
        pass

    def serialize(self, serial_dir: str):
        data = {
            "type": type(self).__name__,
        }
        return yaml.dump(data, default_flow_style=False)

    def load(self, filepath:str, params:Dict):
        pass