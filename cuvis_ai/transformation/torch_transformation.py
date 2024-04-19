from typing import Optional, Any, Dict
import yaml
import pickle as pk
import torch
import os
from . import BaseTransformation

class TorchTransformation(BaseTransformation):
    """Node representing a simple mathematical transformation of data using a pytorch function.

    Args:
        function_name: The name of the pytorch function to use. Almost any function available from the torch module should work.
        operand_b: A constant value to pass into the function alongside the regular input data.
        kwargs: Any additional keyword arguments will be passed to the pytorch function anytime it is called.
    """
        
    def __init__(self, function_name: Optional[str]=None, *, operand_b: Optional[Any]=None, **kwargs):
        super().__init__()
        self.op_name = function_name
        self.b = operand_b
        self.input_size_y = None
        self.fun_kwargs = kwargs
            
        if self.op_name is not None:
            self.initialized = True
            self.fun = getattr(torch, self.op_name)
        else:
            self.fun = None
            self.initialized = False

    def forward(self, X: Any, Y: Optional[Any]=None):
        try:
            if Y is not None:
                return self.fun(X, Y, **self.fun_kwargs)
            elif self.b is not None:
                return self.fun(X, self.b, **self.fun_kwargs)
            else:
                return self.fun(X, **self.fun_kwargs)
        except RuntimeError as re:
            raise RuntimeError(F"TorchTransformation with operation '{self.op_name}' was called with non-matching input and " \
                             F"{'constant ' if self.b is not None else ''}second operand shapes!\nPyTorch reports: '{re}'")

    def fit(self, X: Any, Y: Optional[Any]=None):
        if Y is not None and self.b is not None:
            raise ValueError(F"TorchTransformation with operation '{self.op_name}' was given a constant value and a second operand!" \
                             "\nTorchTransformation can have none or one of either, but must not have both.")
        self.input_size = X.shape
        if Y is not None:
            self.input_size_y = Y.shape
        try:
            self.output_size = self.forward(X, Y if Y is not None else self.b).shape
        except RuntimeError as re:
            raise ValueError(F"TorchTransformation with operation '{self.op_name}' has non-matching input and " \
                             F"{'constant ' if self.b is not None else ''}second operand shapes!\nPyTorch reports: '{re}'")
        
    def check_output_dim(self, X: Any, Y: Optional[Any]=None):
        out_size = self.forward(X, Y)
        assert(out_size == self.output_size)

    def check_input_dim(self, X: Any, Y: Optional[Any]=None):
        try:
            self.forward(X, Y)
        except RuntimeError:
            assert(False)

    def serialize(self, serial_dir: str):
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return

        blob = (self.b, self.fun_kwargs)
        blobfile_path = os.path.join(serial_dir, F"{hash(str(blob))}_torchtransformation.pkl")
        with open(blobfile_path, "wb") as blobfile:
            pk.dump(blob, blobfile)
        
        data = {
            "type": type(self).__name__,
            "op_name": self.op_name,
            "transformation_blob": blobfile_path,
            "input_size_x": self.input_size,
            "input_size_y": self.input_size_y,
        }
        return yaml.dump(data, default_flow_style=False)

    def load(self, filepath:str, params:Dict):
        blobfile_path = os.path.join(filepath, params.get("transformation_blob"))
        with open(blobfile_path, "rb") as blobfile:
            self.b, self.fun_kwargs = pk.load(blobfile)
        self.op_name = params.get("op_name")
        self.fun = getattr(torch, self.op_name)
        self.input_size = params.get("input_size_x")
        self.input_size_y = params.get("input_size_y")
        self.initialized = True