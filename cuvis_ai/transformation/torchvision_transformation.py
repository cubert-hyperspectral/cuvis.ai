
import torch

import numpy as np
from typing import Optional, Any, Dict, Callable, Tuple, Union, Iterable
from ..node.base import BaseTransformation
from ..node import Node
from pathlib import Path


class TorchVisionTransformation(Node, BaseTransformation):
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

    def __init__(self, tv_transform: Optional[Callable] = None):
        super().__init__()
        self.tv_transform = tv_transform
        self.initialized = self.tv_transform is not None

    def forward(self, X: np.ndarray) -> Any:
        """ Transform data and labels according to the torchvision transform this node represents.
        Parameters
        ----------
        X : np.ndarray
            Expects a numpy array or torch tensor (data, [labels, [metadata]]) as returned by dataloaders or just a single tensor.
        Returns
        -------
        Tuple
            The transformed data including any labels and meta-data passed in, if any.
        """

        if isinstance(X, np.ndarray):
            return self.tv_transform(torch.as_tensor(X).permute([0, 3, 1, 2])).permute([0, 2, 3, 1]).numpy()

    def fit(self, X: Union[Tuple, np.ndarray]):
        pass

    @Node.output_dim.getter
    def output_dim(self) -> Tuple[int, int, int]:
        return (-1, -1, -1)

    @Node.input_dim.getter
    def input_dim(self) -> Tuple[int, int, int]:
        return (-1, -1, -1)

    def serialize(self, serial_dir: str):
        """Serialize this node."""
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return

        blobfile_name = F"{hash(self.tv_transform)}_tvtransformation.zip"
        torch.save(self.tv_transform, Path(serial_dir) / blobfile_name)

        data = {
            'id': self.id,
            "type": type(self).__name__,
            "tv_transform": blobfile_name,
        }
        return data

    def load(self, params: dict, serial_dir: str):
        """Load this node from a serialized graph."""
        self.id = params.get('id')
        self.tv_transform = torch.load(
            Path(serial_dir) / params.get("tv_transform"))
        self.initialized = True
