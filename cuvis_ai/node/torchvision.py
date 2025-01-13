
import numpy as np
from typing import Any, Union

import torchvision.transforms.v2
from .base import BaseTransformation
from . import Node
import torchvision
import torch


def _wrap_torchvision_transform(cls):

    class WrappedTorchVisionTransformation(Node, BaseTransformation):

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.tv_transform = cls(*args, **kwargs)
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

        def fit(self, X: Union[tuple, np.ndarray]):
            pass

        @Node.output_dim.getter
        def output_dim(self) -> tuple[int, int, int]:
            return (-1, -1, -1)

        @Node.input_dim.getter
        def input_dim(self) -> tuple[int, int, int]:
            return (-1, -1, -1)

        def serialize(self, serial_dir: str) -> dict[str, Any]:
            """Serialize this node."""
            if not self.initialized:
                print('Module not fully initialized, skipping output!')
                return dict()

            data_independent = self.tv_transform.make_params()
            return {'params': data_independent}

        def load(self, params: dict, serial_dir: str):
            """Load this node from a serialized graph."""
            self.id = params.get('id')
            self.tv_transform = cls(**params['params'])
            self.initialized = True


def _wrap_torchvision_class(cls):
    if issubclass(cls, torchvision.transforms.v2.Transform):
        return _wrap_torchvision_transform(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_torchvision_instance(obj):
    cls = _wrap_torchvision_class(obj.__class__)

    params = obj.get_params()

    return cls(**params)
