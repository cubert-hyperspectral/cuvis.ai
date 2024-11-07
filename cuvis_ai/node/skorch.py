
import functools
import torch

from ..utils.numpy import *
from ..utils.torch import InputDimension, guess_input_dimensionalty
from .node import Node
from .base import BaseSupervised, BaseUnsupervised

import numpy as np

import torch
import torch.nn as nn
import skorch


def _wrap_preprocessor_class(cls):
    pass


def _wrap_supervised_class(cls):

    class SkorchWrappedSupervised(Node, BaseSupervised):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        def __init__(self, *args, criterion=torch.nn.NLLLoss, **kwargs):
            super().__init__()
            self.input_size = (-1, -1, -1)
            self.output_size = (-1, -1, -1)

            self.model_args = {f'module__{k}': v for k,
                               v in kwargs.items()}

            self.model_args_no_prefix = {k: v for k,
                                         v in kwargs.items()}

            self.criterion = criterion

            self.expected_dim = guess_input_dimensionalty(
                cls(**self.model_args_no_prefix))

            self.net = skorch.NeuralNetClassifier(
                module=cls,
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
            if self.expected_dim == InputDimension.One:
                flattened_data = flatten_batch_and_spatial(X)
                flattened_label = flatten_labels(Y)
            elif self.expected_dim == InputDimension.Two:
                flattened_data = flatten_spatial(X)
                flattened_label = flatten_labels(Y)

            self.net.fit(flattened_data, flattened_label)

            self.input_size = (-1, -1, self.n_features_in_)
            self.output_size = (-1, -1, self._n_features_out)
            self.initialized = True

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = self.net.predict(self, flattened_data)
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

    SkorchWrappedSupervised.__name__ = cls.__name__
    functools.update_wrapper(SkorchWrappedSupervised.__init__, cls.__init__)
    return SkorchWrappedSupervised


def _wrap_unsupervised_class(cls):

    class SkorchWrappedUnsupervised(Node, BaseUnsupervised):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        def __init__(self, *args, criterion, **kwargs):
            super().__init__()
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)

            self.model_args = {f'module__{k}': v for k,
                               v in kwargs.items()}

            self.model_args_no_prefix = {k: v for k,
                                         v in kwargs.items()}

            self.criterion = criterion

            self.expected_dim = guess_input_dimensionalty(
                cls(**self.model_args_no_prefix))

            self.net = skorch.NeuralNet(
                module=cls,
                criterion=self.criterion,
                **self.model_args
            )

            self.initialized = False

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        def fit(self, X: np.ndarray):

            if self.expected_dim == InputDimension.One:
                flattened_data = flatten_batch_and_spatial(X)
            elif self.expected_dim == InputDimension.Three:
                flattened_data = np.moveaxis(X, -1, -3)

            # y can be set to None, in that case it will be derived from X
            self.net.fit(flattened_data, flattened_data)

            self._input_size = X.shape
            self._output_size = X.shape
            self.initialized = True

        def forward(self, X: np.ndarray):
            if self.expected_dim == InputDimension.One:
                flattened_data = flatten_batch_and_spatial(X)
            elif self.expected_dim == InputDimension.Three:
                flattened_data = np.moveaxis(X, -1, -3)
            transformed_data = self.net.predict(flattened_data)
            if self.expected_dim == InputDimension.One:
                return unflatten_batch_and_spatial(transformed_data, X.shape)
            elif self.expected_dim == InputDimension.Three:
                return np.moveaxis(X, -1, -3)

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

    SkorchWrappedUnsupervised.__name__ = cls.__name__
    functools.update_wrapper(SkorchWrappedUnsupervised.__init__, cls.__init__)
    return SkorchWrappedUnsupervised


def _wrap_torch_class(cls):
    if issubclass(cls, nn.Module):
        return _wrap_unsupervised_class(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_torch_instance(obj):
    cls = _wrap_torch_class(obj.__class__)

    params = obj.get_params()

    return cls(**params)
