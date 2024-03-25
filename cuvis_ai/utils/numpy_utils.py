
import numpy as np




def flatten_arrays(data: np.ndarray):
    dims = len(data.shape)
    orig_shape = data.shape
    if dims  == 2:
        # list of channels
        return data
    if dims == 3:
        n_pixels = data.shape[1] * data.shape[2]
        spectra_list = data.reshape((data.shape[0],n_pixels))
        return spectra_list, orig_shape

    if dims == 4:
        #flatten_data = data.reshape(-1, *data.shape[-2:])
        n_pixels = data.shape[0] * data.shape[2] * data.shape[3]
        spectra_list = data.reshape((n_pixels, data.shape[1]))
        # TODO the data is most likely falsely rearranged
        return spectra_list, orig_shape
    
def unflatten_arrays(data: np.ndarray, orig_shape):
    return data.reshape(orig_shape)


def flatten_labels(data: np.ndarray):
    dims = len(data.shape)
    orig_shape = data.shape
    if dims  == 2:
        n_pixels = data.shape[0] * data.shape[1]
        label_list = data.reshape(n_pixels)
        return label_list, orig_shape
    if dims == 3:
        n_pixels = data.shape[0] * data.shape[1] * data.shape[2]
        label_list = data.reshape(n_pixels)
        return label_list, orig_shape