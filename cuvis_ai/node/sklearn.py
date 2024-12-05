

import functools

from ..utils.numpy import flatten_batch_and_spatial, unflatten_batch_and_spatial, flatten_batch_and_labels
from .node import Node
import uuid

import numpy as np
from pathlib import Path

from sklearn.base import TransformerMixin, ClassifierMixin, ClusterMixin, DensityMixin
from .base import Preprocessor, BaseUnsupervised, BaseSupervised


class SklearnWrapped:
    pass


def _serialize_sklearn_model(obj, cls, data_dir: Path) -> dict:
    data_independent = cls.get_params(obj)
    if not obj.initialized:
        return {'params': data_independent}

    def ignore_exceptions(obj, attr):
        try:
            getattr(obj, attr)
            return True
        except:
            return False

    data_dependend = {
        attr: getattr(obj, attr)
        for attr in dir(obj)
        if attr.endswith("_")
        and ignore_exceptions(obj, attr)
        and not callable(getattr(obj, attr))
        and not attr.startswith("__")
        and not attr[:-1] in data_independent.keys()
    }
    return {'params': data_independent, 'state': data_dependend}


def _load_sklearn_model(obj, cls, params: dict, data_dir: Path) -> None:
    data_independent_keys = set(cls.get_params(obj).keys())

    params_independent = {key: params['params'][key]
                          for key in data_independent_keys}

    cls.set_params(obj, **params_independent)

    if 'state' not in params.keys():
        return

    data_dependent_keys = {
        key for key in params['state'].keys()}

    params_dependent = {key: params['state'][key]
                        for key in data_dependent_keys}

    for k, v in params_dependent.items():
        try:
            setattr(obj, k, v)
        except:
            print(f'Could not set state attribute {k} for {obj.id}')  # nopep8
    obj.initialized = True
    obj._derive_values()


def _wrap_preprocessor_class(cls):

    class SklearnWrappedPreprocessor(Node, Preprocessor, SklearnWrapped):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            Node.__init__(self)
            self.id = f'{cls.__name__}-{str(uuid.uuid4())}'
            self._wrapped = cls(*args, **kwargs)
            __name__ = cls.__name__
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        def fit(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            self._wrapped.fit(flattened_data)
            self.initialized = True
            self._derive_values()

        def _derive_values(self):
            if not self.initialized:
                return
            self._input_size = (-1, -1, self._wrapped.n_features_in_)
            self._output_size = (-1, -1, self._wrapped._n_features_out)

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = self._wrapped.transform(flattened_data)
            return unflatten_batch_and_spatial(transformed_data, X.shape)

        def serialize(self, data_dir: Path) -> dict:
            return _serialize_sklearn_model(self._wrapped, cls, data_dir)

        def load(self, params: dict, data_dir: Path) -> None:
            return _load_sklearn_model(self._wrapped, cls, params, data_dir)

    SklearnWrappedPreprocessor.__name__ = cls.__name__
    functools.update_wrapper(SklearnWrappedPreprocessor.__init__, cls.__init__)
    return SklearnWrappedPreprocessor


def _wrap_supervised_class(cls):

    class SklearnWrappedSupervised(Node, BaseSupervised, SklearnWrapped):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            Node.__init__(self)
            self.id = f'{cls.__name__}-{str(uuid.uuid4())}'
            self._wrapped = cls(*args, **kwargs)
            __name__ = cls.__name__
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        def fit(self, X: np.ndarray, Y: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            flattened_label = flatten_batch_and_labels(Y)
            self._wrapped.fit(flattened_data, flattened_label)
            self.initialized = True
            self._derive_values()

        def _derive_values(self):
            if not self.initialized:
                return
            self._input_size = (-1, -1, self._wrapped.n_features_in_)
            self._output_size = (-1, -1, 1)

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            transformed_data = self._wrapped.predict_proba(flattened_data)
            return unflatten_batch_and_spatial(transformed_data, X.shape)

        def serialize(self, data_dir: Path) -> dict:
            return _serialize_sklearn_model(self._wrapped, cls, data_dir)

        def load(self, params: dict, data_dir: Path) -> None:
            return _load_sklearn_model(self._wrapped, cls, params, data_dir)

    SklearnWrappedSupervised.__name__ = cls.__name__
    functools.update_wrapper(SklearnWrappedSupervised.__init__, cls.__init__)
    return SklearnWrappedSupervised


def _wrap_unsupervised_class(cls):

    class SklearnWrappedUnsupervised(Node, BaseUnsupervised, SklearnWrapped):

        __doc__ = cls.__doc__
        __module__ = cls.__module__

        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            Node.__init__(self)
            self.id = f'{cls.__name__}-{str(uuid.uuid4())}'
            self._wrapped = cls(*args, **kwargs)
            __name__ = cls.__name__
            self._input_size = (-1, -1, -1)
            self._output_size = (-1, -1, -1)

        @Node.input_dim.getter
        def input_dim(self):
            return self._input_size

        @Node.output_dim.getter
        def output_dim(self):
            return self._output_size

        def fit(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            self._wrapped.fit(flattened_data)
            self.initialized = True
            self._derive_values()

        def _derive_values(self):
            if not self.initialized:
                return
            self._input_size = (-1, -1, self._wrapped.n_features_in_)
            self._output_size = (-1, -1, 1)

        def forward(self, X: np.ndarray):
            flattened_data = flatten_batch_and_spatial(X)
            prediction_data = self._wrapped.predict_proba(flattened_data)
            return unflatten_batch_and_spatial(prediction_data, X.shape)

        def serialize(self, data_dir: Path) -> dict:
            return _serialize_sklearn_model(self._wrapped, cls, data_dir)

        def load(self, params: dict, data_dir: Path) -> None:
            return _load_sklearn_model(self._wrapped, cls, params, data_dir)

    SklearnWrappedUnsupervised.__name__ = cls.__name__
    functools.update_wrapper(SklearnWrappedUnsupervised.__init__, cls.__init__)
    return SklearnWrappedUnsupervised


def _wrap_sklearn_class(cls):
    if issubclass(cls, ClusterMixin):
        return _wrap_unsupervised_class(cls)
    elif issubclass(cls, DensityMixin):
        return _wrap_unsupervised_class(cls)
    elif issubclass(cls, ClassifierMixin):
        return _wrap_supervised_class(cls)
    elif issubclass(cls, TransformerMixin):
        return _wrap_preprocessor_class(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_sklearn_instance(obj):
    cls = _wrap_sklearn_class(obj.__class__)

    params = obj.get_params()

    return cls(**params)
