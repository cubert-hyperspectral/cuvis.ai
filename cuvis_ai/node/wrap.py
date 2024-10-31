
from types import MethodWrapperType, ModuleType
import inspect
import functools
import torch

from sklearn.base import BaseEstimator

from .sklearn import _wrap_sklearn_class, _wrap_sklearn_instance


def _wrap_class(cls):

    if issubclass(cls, BaseEstimator):
        return _wrap_sklearn_class(cls)
    else:
        raise ValueError("Called on unsupported class")


def _wrap_instance(obj):

    if isinstance(obj, BaseEstimator):
        return _wrap_sklearn_instance(obj)
    else:
        raise ValueError("Called on unsupported object")


def node(wrapped):
    """Node Wrapper / Decorator. Use to wrap a specific module into a node."""

    if isinstance(wrapped, ModuleType):
        raise NotImplementedError('Currently cannot be wrapped')

    if inspect.isclass(wrapped):
        return _wrap_class(wrapped)
    if isinstance(wrapped, object):
        return _wrap_instance(wrapped)
