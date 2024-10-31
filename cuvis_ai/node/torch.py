

from types import MethodWrapperType, ModuleType
import inspect
import functools
import torch
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union, Any
from collections import Counter, namedtuple, OrderedDict

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial, flatten_batch_and_labels
from ..utils.numpy_utils import flatten_spatial, flatten_labels, unflatten_spatial
from .node import Node
import uuid

import numpy as np

import torch
import torch.nn as nn
import skorch


def _wrap_preprocessor_class(cls):
    pass


def _wrap_supervised_class(cls):

    class SkorchWrappedSupervised(Node):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        def __init__(self, *args, criterion=torch.nn.NLLLoss, **kwargs):
            self.id = f'{cls.__name__}-{str(uuid.uuid4())}'
            self.input_size = (-1, -1, -1)
            self.output_size = (-1, -1, -1)

            self.model_args = {f'module__{k}': v for k,
                               v in kwargs.items()}

            self.model_args_no_prefix = {k: v for k,
                                         v in kwargs.items()}

            self.criterion = criterion

            self.net = skorch.NeuralNetClassifier(
                module=cls,
                module__num_units=100,
                module__dropout=0.5,
                criterion=self.criterion,
            )

            self.initialized = False

        @Node.input_dim.getter
        def input_dim(self):
            return self.input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self.output_size

        def fit(self, X: np.ndarray, Y: np.ndarray):
            flattened_data = flatten_spatial(X)
            flattened_label = flatten_labels(Y)

            self.net.fit(self, flattened_data)

            self.input_size = (-1, -1, self.n_features_in_)
            self.output_size = (-1, -1, self._n_features_out)
            self.initialized = True

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = self.net.transform(self, flattened_data)
            return unflatten_batch_and_spatial(transformed_data, X.shape)

        def serialize(self) -> dict:
            data_independent = self.net.get_params(self)
            data_dependend = {
                attr: getattr(self, attr)
                for attr in dir(self)
                if attr.endswith("_") and not callable(getattr(self, attr)) and not attr.startswith("__")
            }
            return data_independent | data_dependend

        def load(self, params: dict) -> None:
            data_independent_keys = set(self.net.get_params(self).keys())

            data_dependent_keys = {
                key for key in params.keys() if key not in data_independent_keys and key.endswith("_")}

            params_independent = {key: params[key]
                                  for key in data_independent_keys}

            self.net.set_params(self, **params_independent)

            params_dependent = {key: params[key]
                                for key in data_dependent_keys}

            for k, v in params_dependent.items():
                setattr(self, k, v)

        def __repr__(self):
            return self.net.__repr__()

    functools.update_wrapper(SkorchWrappedSupervised.__init__, cls.__init__)
    return SkorchWrappedSupervised


def _wrap_torch_class(cls):
    if issubclass(cls, nn.Module):
        return _wrap_supervised_class(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_torch_instance(obj):
    cls = _wrap_torch_class(obj.__class__)

    params = obj.get_params()

    return cls(**params)
