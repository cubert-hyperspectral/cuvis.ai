
import numpy as np
from typing import Tuple, Union

def get_shape_without_batch(array: np.ndarray, ignore = []):
    ndim = array.ndim
    if ndim != 3 and ndim != 4:
        raise ValueError("Input array must be 3D or 4D.")
    shape =  array.shape if ndim == 3 else array.shape[1:]
    shape = [-1 if i in ignore else shape[i] for i in [0,1,2]]
    return shape

    
def check_array_shape(array: Union[np.ndarray, Tuple[int,int,int]], wanted_shape: Tuple[int,int,int]):
    if isinstance(array, np.ndarray):
        array_shape = array.shape
    else:
        array_shape = array

    ret = True
    for v, w in zip(array_shape, wanted_shape):
        ret &= w == -1 or v == w
    return ret

def flatten_spatial(array: np.ndarray):
    if array.ndim == 3:
        # Array is of shape [width, height, channels]
        return array.reshape(-1, array.shape[2])
    elif array.ndim == 4:
        # Array is of shape [batch, width, height, channels]
        return array.reshape(array.shape[0], -1, array.shape[3])
    else:
        raise ValueError("Input array must be 3D or 4D.")
    
def flatten_batch_and_spatial(array: np.ndarray):
    if array.ndim == 3:
        # Array is of shape [width, height, channels]
        return array.reshape(-1, array.shape[2])
    elif array.ndim == 4:
        # Array is of shape [batch, width, height, channels]
        return array.reshape(-1, array.shape[3])
    else:
        raise ValueError("Input array must be 3D or 4D.")
    
def unflatten_batch_and_spatial(array: np.ndarray, orig_shape):
    if array.ndim != 2:
        raise ValueError("Input array must be 2D or 3D.")
    if array.shape[0] != np.prod(orig_shape[:-1]):
        raise ValueError("Input array and orig shape do not add up.")
    return array.reshape(*orig_shape[:-1],-1)

def unflatten_spatial(array: np.ndarray, orig_shape):
    if array.ndim != 3 and array.ndim != 2:
        raise ValueError("Input array must be 2D or 3D.")
    if array.shape[0] != np.prod(orig_shape[:-1]):
        raise ValueError("Input array and orig shape do not add up.")
    return array.reshape(*orig_shape[:-1],-1)

def flatten_labels(array: np.ndarray):
    if array.ndim == 2:
        # Array is of shape [width, height]
        return array.reshape(-1)
    elif array.ndim == 3:
        # Array is of shape [batch, width, height]
        return array.reshape(array.shape[0], -1)
    else:
        raise ValueError("Input array must be 2D or 3D.")

def unflatten_labels(array: np.ndarray, orig_shape):
    if array.ndim != 1 and array.ndim != 2:
        raise ValueError("Input array must be 1D or 2D.")
    return array.reshape(orig_shape)

def flatten_batch_and_labels(array: np.ndarray):
    if array.ndim == 2:
        # Array is of shape [width, height]
        return array.reshape(-1)
    elif array.ndim == 3:
        # Array is of shape [batch, width, height]
        return array.reshape(-1)
    else:
        raise ValueError("Input array must be 2D or 3D.")

def unflatten_batch_and_labels(array: np.ndarray, orig_shape):
    if array.ndim != 1:
        raise ValueError("Input array must be 1D.")
    return array.reshape(orig_shape)
