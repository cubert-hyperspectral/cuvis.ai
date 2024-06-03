from typing import Optional, Any, Dict, Callable, Tuple
import yaml
import torch
import os
from . import BaseTransformation

class TorchVisionTransformation(BaseTransformation):
    """ Node for applying a torchvision transform within the pipeline.
    For proper functionality, these transformations should be added to the dataloader.
    Any transform present in torchvision.transforms.v2 should be compatible.
    
    Parameters
    ----------
    tv_transform : Callable,optional
        The transform that this node should represent. Can also be a composition.

    Notes
    -----
    Torchvision Transformations added to the graph using this node only apply the HSI data (the cube)!
    Only transformations added to the dataloader apply to labels and metadata as well.
    """
    
    def __init__(self, tv_transform: Optional[Callable]=None):
        super().__init__()
        self.id = F"{self.__class__.__name__}-{str(uuid.uuid4())}"
        self.tv_transform = tv_transform
        self.initialized = self.tv_transform is not None

    def forward(self, X: Union[Tuple, np.ndarray]) -> Any:
        """ Transform data and labels according to the torchvision transform this node represents.
        Parameters
        ----------
        X : Tuple
            Expects a tuple (data, [labels, [metadata]]) as returned by dataloaders or just a single tensor.
        Returns
        -------
        Tuple
            The transformed data including any labels and meta-data passed in, if any.
        """
        
        if isinstance(X, tuple):
            cube = self.tv_transform(torch.as_tensor(X[0]).permute([0, 3, 1, 2])).permute([0, 2, 3, 1]).numpy()
            if len(X) > 1:
                return (cube, X[1:])
            return cube
        elif isinstance(X, np.ndarray):
            return self.tv_transform(torch.as_tensor(X).permute([0, 3, 1, 2])).permute([0, 2, 3, 1]).numpy()
        else:
            raise ValueError(F"TorchVisionTransformation expected tuple or numpy array but got {type(X)}!")

    def fit(self, X: Union[Tuple, np.ndarray]):
        pass
        
    def check_output_dim(self, X: Iterable):
        pass

    def check_input_dim(self, X: Iterable):
        pass

    def serialize(self, serial_dir: str):
        """Serialize this node."""
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return

        blobfile_path = os.path.join(serial_dir, F"{hash(self.tv_transform)}_tvtransformation.zip")
        torch.save(self.tv_transform, blobfile_path)
        
        data = {
            "type": type(self).__name__,
            "tv_transform": blobfile_path,
        }
        return yaml.dump(data, default_flow_style=False)

    def load(self, filepath:str, params:Dict):
        """Load this node from a serialized graph."""
        blobfile_path = os.path.join(filepath, params.get("tv_transform"))
        self.tv_transform = torch.load(blobfile_path)
        self.initialized = True