
import numpy as np


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
    return array.reshape(orig_shape)

def unflatten_spatial(array: np.ndarray, orig_shape):
    if array.ndim != 3 and array.ndim != 2:
        raise ValueError("Input array must be 2D or 3D.")
    return array.reshape(orig_shape)

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

def unflatten_labels(array: np.ndarray, orig_shape):
    if array.ndim != 1:
        raise ValueError("Input array must be 1D.")
    return array.reshape(orig_shape)
