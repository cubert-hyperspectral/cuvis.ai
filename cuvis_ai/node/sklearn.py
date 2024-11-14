

import functools

from ..utils.numpy import flatten_batch_and_spatial, unflatten_batch_and_spatial, flatten_batch_and_labels
from .node import Node
import uuid

import numpy as np
from pathlib import Path

from sklearn.base import TransformerMixin, ClassifierMixin, ClusterMixin
from .base import Preprocessor


def _wrap_preprocessor_class(cls):

    class SklearnWrappedPreprocessor(cls, Node, Preprocessor):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            self.id = f'{cls.__name__}-{str(uuid.uuid4())}'
            cls.__init__(self, *args, **kwargs)
            __name__ = cls.__name__
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)
            self.initialized = False

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        def fit(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            cls.fit(self, flattened_data)
            self.initialized = True
            self._derive_values()

        def _derive_values(self):
            if not self.initialized:
                return
            self._input_size = (-1, -1, self.n_features_in_)
            self._output_size = (-1, -1, self._n_features_out)

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = cls.transform(self, flattened_data)
            return unflatten_batch_and_spatial(transformed_data, X.shape)

        def serialize(self, data_dir: Path) -> dict:
            data_independent = cls.get_params(self)
            if not self.initialized:
                return {'params': data_independent}
            data_dependend = {
                attr: getattr(self, attr)
                for attr in dir(self)
                if attr.endswith("_")
                and not callable(getattr(self, attr))
                and not attr.startswith("__")
                and not attr[:-1] in data_independent.keys()
            }
            # data_dependend['_n_features_out'] = self._n_features_out
            return {'params': data_independent, 'state': data_dependend}

        def load(self, params: dict, data_dir: Path) -> None:
            data_independent_keys = set(cls.get_params(self).keys())

            params_independent = {key: params['params'][key]
                                  for key in data_independent_keys}

            cls.set_params(self, **params_independent)

            if 'state' not in params.keys():
                return

            data_dependent_keys = {
                key for key in params['state'].keys()}

            params_dependent = {key: params['state'][key]
                                for key in data_dependent_keys}

            for k, v in params_dependent.items():
                setattr(self, k, v)
            self.initialized = True
            self._derive_values()

    SklearnWrappedPreprocessor.__name__ = cls.__name__
    functools.update_wrapper(SklearnWrappedPreprocessor.__init__, cls.__init__)
    return SklearnWrappedPreprocessor


def _wrap_supervised_class(cls):

    class SklearnWrappedSupervised(cls, Node):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            self.id = f'{cls.__name__}-{str(uuid.uuid4())}'
            cls.__init__(self, *args, **kwargs)
            __name__ = cls.__name__
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)
            self.initialized = False

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        def fit(self, X: np.ndarray, Y: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            flattened_label = flatten_batch_and_labels(Y)
            cls.fit(self, flattened_data, flattened_label)
            self.initialized = True
            self._derive_values()

        def _derive_values(self):
            if not self.initialized:
                return
            self._input_size = (-1, -1, self.n_features_in_)
            self._output_size = (-1, -1, 1)

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = cls.transform(self, flattened_data)
            return unflatten_batch_and_spatial(transformed_data, X.shape)

        def serialize(self, data_dir: Path) -> dict:
            data_independent = cls.get_params(self)
            if not self.initialized:
                return {'params': data_independent}
            data_dependend = {
                attr: getattr(self, attr)
                for attr in dir(self)
                if attr.endswith("_")
                and not callable(getattr(self, attr))
                and not attr.startswith("__")
                and not attr[:-1] in data_independent.keys()
            }
            return {'params': data_independent, 'state': data_dependend}

        def load(self, params: dict, data_dir: Path) -> None:
            data_independent_keys = set(cls.get_params(self).keys())

            params_independent = {key: params['params'][key]
                                  for key in data_independent_keys}

            cls.set_params(self, **params_independent)

            if 'state' not in params.keys():
                return

            data_dependent_keys = {
                key for key in params['state'].keys()}

            params_dependent = {key: params['state'][key]
                                for key in data_dependent_keys}

            for k, v in params_dependent.items():
                setattr(self, k, v)
            self.initialized = True
            self._derive_values()

    SklearnWrappedSupervised.__name__ = cls.__name__
    functools.update_wrapper(SklearnWrappedSupervised.__init__, cls.__init__)
    return SklearnWrappedSupervised


def _wrap_unsupervised_class(cls):

    class SklearnWrappedUnsupervised(cls, Node):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            self.id = f'{cls.__name__}-{str(uuid.uuid4())}'
            cls.__init__(self, *args, **kwargs)
            __name__ = cls.__name__
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)
            self.initialized = False

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        def fit(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            cls.fit(self, flattened_data)
            self.initialized = True
            self._derive_values()

        def _derive_values(self):
            if not self.initialized:
                return
            self._input_size = (-1, -1, self.n_features_in_)
            self._output_size = (-1, -1, 1)

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            prediction_data = cls.predict(self, flattened_data)
            return unflatten_batch_and_spatial(prediction_data, X.shape)

        def serialize(self, data_dir: Path) -> dict:
            data_independent = cls.get_params(self)
            if not self.initialized:
                return {'params': data_independent}
            data_dependend = {
                attr: getattr(self, attr)
                for attr in dir(self)
                if attr.endswith("_")
                and not callable(getattr(self, attr))
                and not attr.startswith("__")
                and not attr[:-1] in data_independent.keys()
            }
            return {'params': data_independent, 'state': data_dependend}

        def load(self, params: dict, data_dir: Path) -> None:
            data_independent_keys = set(cls.get_params(self).keys())

            params_independent = {key: params['params'][key]
                                  for key in data_independent_keys}

            cls.set_params(self, **params_independent)

            if 'state' not in params.keys():
                return

            data_dependent_keys = {
                key for key in params['state'].keys()}

            params_dependent = {key: params['state'][key]
                                for key in data_dependent_keys}

            for k, v in params_dependent.items():
                setattr(self, k, v)
            self.initialized = True
            self._derive_values()

    SklearnWrappedUnsupervised.__name__ = cls.__name__
    functools.update_wrapper(SklearnWrappedUnsupervised.__init__, cls.__init__)
    return SklearnWrappedUnsupervised


def _wrap_sklearn_class(cls):
    if issubclass(cls, ClusterMixin):
        return _wrap_unsupervised_class(cls)
    elif issubclass(cls, TransformerMixin):
        return _wrap_preprocessor_class(cls)
    elif issubclass(cls, ClassifierMixin):
        return _wrap_supervised_class(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_sklearn_instance(obj):
    cls = _wrap_sklearn_class(obj.__class__)

    params = obj.get_params()

    return cls(**params)