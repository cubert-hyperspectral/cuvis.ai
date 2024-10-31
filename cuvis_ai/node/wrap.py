
from types import MethodWrapperType, ModuleType
import inspect
import functools
import torch
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union, Any
from collections import Counter, namedtuple, OrderedDict

from ..utils.numpy_utils import flatten_batch_and_spatial, unflatten_batch_and_spatial
from .node import Node

import numpy as np


def get_np_dummy_data(shape):
    return np.random.rand(*shape)


class SklearnNode(Node):
    def __init__(self, wrapped):
        self.wrapped = wrapped


def _wrap_class(cls):

    class SklearnWrapped(cls):

        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)

        def fit(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)

            cls.fit(self, flattened_data)

            self.input_size = (-1, -1, self.n_features_in_)
            self.output_size = (-1, -1, self._n_features_out)
            self.initialized = True

        def forward(self, X):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = cls.transform(self, flattened_data)
            return unflatten_batch_and_spatial(transformed_data, X.shape)

        def serialize(self) -> dict:
            data_independent = cls.get_params(self)
            data_dependend = {
                attr: getattr(self, attr)
                for attr in dir(self)
                if attr.endswith("_") and not callable(getattr(self, attr)) and not attr.startswith("__")
            }
            return data_independent | data_dependend

        def load(self, params: dict) -> None:
            data_independent_keys = set(cls.get_params(self).keys())

            data_dependent_keys = {
                key for key in params.keys() if key not in data_independent_keys}

            params_independent = {key: params[key]
                                  for key in data_independent_keys}

            cls.set_params(self, **params_independent)

            params_dependent = {key: params[key]
                                for key in data_dependent_keys}

            for k, v in params_dependent.items():
                setattr(self, k, v)

    return SklearnWrapped


def _wrap_class_instance(obj):

    return obj


def node(wrapped):
    """Node Wrapper / Decorator. Use to wrap a specific module into a node."""

    if isinstance(wrapped, ModuleType):
        raise NotImplementedError('Currently cannot be wrapped')

    if inspect.isclass(wrapped):
        return _wrap_class(wrapped)
    if isinstance(wrapped, object):
        return _wrap_class_instance(wrapped)
