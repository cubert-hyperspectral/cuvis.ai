from typing import Optional, Any, Dict, Iterable
import yaml
import pickle as pk
import torch
import os
from . import BaseTransformation

class TorchTransformation(BaseTransformation):
    """Node representing a transformation of data using a pytorch function.

    Parameters
    ---------
    function_name : str,optional
        The name of the pytorch function to use. Almost any function available from the torch module can be used.
    operand_b : Any,optional
        A constant value to pass into the function alongside the regular input data.
    kwargs : Dict
        Any additional keyword arguments will be passed to the pytorch function anytime it is called.
    """
        
    def __init__(self, function_name: Optional[str]=None, *, operand_b: Optional[Any]=None, **kwargs):
        super().__init__()
        self.id = F"{self.__class__.__name__}-{str(uuid.uuid4())}"
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

    def forward(self, X: Tuple, Y: Optional[Tuple]=None):
        """Apply the pytorch method :arg:`function_name` on :arg:`X`.
        This node basically runs `torch.<function_name>(X, Y)`.
        
        Parameters
        ----------
        X : Tuple
            The first operand for the pytorch method.
        Y : Tuple, optional
            The second operand for the pytorch method.
        
        Returns
        -------
        Any, Tuple
            Returns the result of the pytorch method and any additional labels or metadata passed along with :arg:`X`
        """
        if Y is not None and self.b is not None:
            raise ValueError(F"TorchTransformation with operation '{self.op_name}' was given a constant value and a second operand!" \
                             "\nTorchTransformation can have none or one of either, but must not have both.")
        x_supplemental = None
        if isinstance(X, np.ndarray):
            x_data = torch.as_tensor(X)
        else:
            x_data = torch.as_tensor(X[0])
            x_supplemental = X[1:]
            
        if isinstance(Y, np.ndarray):
            y_data = torch.as_tensor(Y)
        else:
            y_data = torch.as_tensor(Y[0])
        
        try:
            if Y is not None:
                res = self.fun(x_data, y_data, **self.fun_kwargs).numpy()
            elif self.b is not None:
                res = self.fun(x_data, self.b, **self.fun_kwargs).numpy()
            else:
                res = self.fun(x_data, **self.fun_kwargs).numpy()
        except RuntimeError as re:
            raise RuntimeError(F"TorchTransformation with operation '{self.op_name}' was called with non-matching input and " \
                             F"{'constant ' if self.b is not None else ''}second operand shapes!\nPyTorch reports: '{re}'")
        
        return res if (x_supplemental is None) else (res, *x_supplemental)

    def fit(self, X: Any, Y: Optional[Any]=None):
        pass
        
    def check_output_dim(self, X: Any, Y: Optional[Any]=None):
        pass

    def fit(self, X: Iterable, Y: Optional[Iterable]=None):
        pass
        
    def check_output_dim(self, X: Iterable, Y: Optional[Iterable]=None):
        pass

    def check_input_dim(self, X: Iterable, Y: Optional[Iterable]=None):
        try:
            self.forward(X, Y)
        except RuntimeError:
            assert(False)

    def serialize(self, serial_dir: str):
        """Serialize this node and save to :arg:`serial_dir`."""
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
        """Load this node from a serialized graph."""
        blobfile_path = os.path.join(filepath, params.get("transformation_blob"))
        with open(blobfile_path, "rb") as blobfile:
            self.b, self.fun_kwargs = pk.load(blobfile)
        self.op_name = params.get("op_name")
        self.fun = getattr(torch, self.op_name)
        self.input_size = params.get("input_size_x")
        self.input_size_y = params.get("input_size_y")
        self.initialized = True