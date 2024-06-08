from typing import Dict

C_ATTRIB_LIST = ["shape", "wavelengths_nm", "references", "bit_depth", "integration_time_us", "framerate", "flags", "processing_mode"]

"""
The meta-data dictionary is a collection of meta-data for a data cube or dataset.

This information can be extracted from cubes stored in the cu3s file format automatically.

For all other data, a metadata.yaml file can be placed in the root data path to provide this meta data.
The yaml file can either contain any of the attributes of this class directly or contain one or multiple 'fileset' entries.
Each 'fileset' must contain a 'paths' entry, specifying a list of filepaths that this set of attributes is valid for.

Possible entries
----------------
name : str
    Name of the file that this object describes.
shape : tuple
    Numpy shape describing the cube size as (columns, rows, channels)
wavelengths_nm : list
    A list of wavelengths the cube contains. In the same order as the channels.
bit_depth : int
    Bit depth of the source data the cube was computed from.
references : Dict
    A dictionary containing filenames or links to the data references. e.g: white and dark cubes used to calculate reflectance data.
integration_time_us : float
    The integration time (also exposure time) in microseconds used to record the data.
framerate : float
    For video data. The number of measurements taken per second.
flags : Dict
    Data dependend dictionary. Any additional flags associated with measurements. e.g: Overillumination, dead pixels, missing references or data, bad references, key frames, etc.
processing_mode : str
    The processing mode the data was calculated with.
"""



def metadataInit(filename:Dict, fileset_metadata:Dict) -> Dict:
    meta = {"name": filename}
    try:
        if any(((name in filename) for name in fileset_metadata["fileset"]["paths"])):
            meta = fileset_metadata["fileset"]
    except KeyError:
        try:
            meta = fileset_metadata["fileset"]
        except KeyError:
            meta = fileset_metadata
    return meta

