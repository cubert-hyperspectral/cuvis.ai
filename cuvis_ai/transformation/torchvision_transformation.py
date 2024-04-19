from typing import Optional, Any, Dict, Callable
import yaml
import torch
import os
from . import BaseTransformation

class TorchVisionTransformation(BaseTransformation):
    """ Node for applying a torchvision transform within the pipeline.
    In general, these transformation should be added to the dataloader for optimal performance.
    Any transform present in torchvision.transforms.v2 should be compatible.

    Args:
        tv_transform: The transform that this node should represent. Can also be a composition.
    """
    
    def __init__(self, tv_transform: Optional[Callable]=None):
        super().__init__()
        self.tv_transform = tv_transform
        self.initialized = self.tv_transform is not None

    def forward(self, X: Any):
        """ Transform data and labels according to the torchvision transform this node represents.
        Args:
            X: Expects either a tuple (data, metadata, labels) as returned by dataloaders or just data as a list of tensors or a single tensor.
        """
        if isinstance(X, tuple) and len(X) == 3:
            d, m, l = X
            if isinstance(d, torch.Tensor):
                d = self.tv_transform(d.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
            elif isinstance(d, list):
                d = [self.tv_transform(i.permute([0, 3, 1, 2])).permute([0, 2, 3, 1]) for i in d]
            else:
                raise ValueError(F"TorchVisionTransformation expected list or tensor but got {type(d)}!")
            l = self.tv_transform(l)
            m = self.tv_transform(m)
            return (d, m, l)
        if isinstance(X, torch.Tensor):
            return self.tv_transform(X.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        elif isinstance(X, list):
            return [self.tv_transform(i.permute([0, 3, 1, 2])).permute([0, 2, 3, 1]) for i in X]
        else:
            raise ValueError(F"TorchVisionTransformation expected list or tensor but got {type(X)}!")

    def fit(self, X: Any):
        pass
        
    def check_output_dim(self, X: Any):
        pass

    def check_input_dim(self, X: Any):
        pass

    def serialize(self, serial_dir: str):
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
        blobfile_path = os.path.join(filepath, params.get("tv_transform"))
        self.tv_transform = torch.load(blobfile_path)
        self.initialized = True