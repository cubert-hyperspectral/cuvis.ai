import os
os.environ["CUVIS"] = "/usr/lib/cuvis/"
import cuvis
import numpy as np
import glob
import copy
from typing import Optional, Callable
from torchvision import tv_tensors
from pycocotools.coco import COCO
from .Labels2TV import convert_COCO2TV

from .Metadata import Metadata
from .NumpyData import NumpyData, OutputFormat

debug_enabled = False

class CuvisData(NumpyData):
    """Representation for a set of Cuvis data cubes, their meta-data and labels.
    
    See :class:`NumpyData` for more details.
    
    Args:
        root (str): The absolute or relative path to the directory containing the HSI data.
        transforms (callable, optional): A function/transforms that takes in an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        output_format (OutputFormat): Enum value that controls the output format of the dataset. See :class:`OutputFormat`
        output_lambda (callable, optional): Only used when :attr:`output_format` is set to `CustomFilter`. Before returning data, the full output of the dataset is passed through this function to allow for custom filtering.
        
    Note:
        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """

    class _SessionCubeLoader:
        def __init__(self, path, idx):
            self.path = path
            self.idx = idx
        def __call__(self, to_dtype:np.dtype):
            cube = cuvis.SessionFile(self.path).get_measurement(self.idx).data["cube"].array
            cube = np.moveaxis(cube, -1, 0)
            return tv_tensors.Image(cube.astype(to_dtype))
    
    class _LegacyCubeLoader:
        def __init__(self, path):
            self.path = path
        def __call__(self, to_dtype:np.dtype):
            cube = cuvis.Measurement.load(self.path).data["cube"].array
            cube = np.moveaxis(cube, -1, 0)
            return tv_tensors.Image(cube.astype(to_dtype))
    
    def __init__(self, root: str, 
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Optional[Callable] = None,
    ):
        self._FILE_EXTENSION_SESSION = ".cu3s"
        self._FILE_EXTENSION_LEGACY = ".cu3"
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform, output_format=output_format, output_lambda=output_lambda)
        

    def _load_directory(self, dir_path:str):
        if debug_enabled:
            print("Reading from directory:", dir_path)
        fileset_session = glob.glob(os.path.join(self.root, '**/*' + self._FILE_EXTENSION_SESSION), recursive=True)
        
        fileset_legacy = glob.glob(os.path.join(self.root, '**/*' + self._FILE_EXTENSION_LEGACY), recursive=True)
        
        for cur_path in fileset_session:
            self._load_session_file(cur_path)
        for cur_path in fileset_legacy:
            self._load_legacy_file(cur_path)
            
    def _load_session_file(self, filepath: str):
        if debug_enabled:
            print("Found file:", filepath)
        labelpath = os.path.splitext(filepath)[0] + ".json"

        crt_session = cuvis.SessionFile(filepath)

        cube_count = len(crt_session)
        if debug_enabled:
            print("Session file has", cube_count, "cubes")

        if self.metadata_filepath:
            sess_meta = Metadata(filepath, self.fileset_metadata)
        else:
            sess_meta = Metadata(filepath)
        
        temp_mesu = crt_session.get_measurement(0)
        sess_meta.shape = (temp_mesu.data["cube"].width, temp_mesu.data["cube"].height, temp_mesu.data["cube"].channels)
        canvas_size = (sess_meta.shape[0], sess_meta.shape[1])
        sess_meta.wavelengths_nm = temp_mesu.data["cube"].wavelength
        #sess_meta.framerate = crt_session.fps  #TODO: Fix in SDK and reenable
        
        if os.path.isfile(labelpath):
            coco = COCO(labelpath)
            ids = list(sorted(coco.imgs.keys()))
        
        for idx in range(cube_count):
            cube_path = F"{filepath}:{idx}"
            self.paths.append(cube_path)
            self.cubes.append(CuvisData._SessionCubeLoader(filepath, idx))
            
            meta:Metadata = copy.deepcopy(sess_meta)
            # TODO: Add a way to SDK where only meta data is loaded or make the cube lazy-loadable
            #mesu = crt_session.get_measurement(idx)
            #meta.integration_time_us = int(mesu.integration_time * 1000)
            #meta.flags = {}
            #for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            #    meta.flags[key] = val
            #meta.references = {}
            #for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            #    meta.references[key] = val
                
            self.metas.append(meta)

            l = None
            if coco is not None:
                l = convert_COCO2TV(coco.loadAnns(coco.getAnnIds(ids[idx]))[0], canvas_size)
            self.labels.append(l)

    def _load_legacy_file(self, filepath:str):
        if debug_enabled:
            print("Found file:", filepath)
        self.paths.append(filepath)
        labelpath = os.path.splitext(filepath)[0] + ".json"
        
        if self.metadata_filepath:
            meta = Metadata(filepath, self.fileset_metadata)
        else:
            meta = Metadata(filepath)
        
        mesu = cuvis.Measurement(filepath)
        meta.shape = (mesu.data["cube"].width, mesu.data["cube"].height, mesu.data["cube"].channels)
        meta.wavelengths_nm = mesu.data["cube"].wavelength
        
        l = None
        canvas_size = (meta.shape[0], meta.shape[1])
        if os.path.isfile(labelpath):
            coco = COCO(labelpath)
            l = convert_COCO2TV(coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0])), canvas_size)
        self.labels.append(l)
            
        self.cubes.append(CuvisData._LegacyCubeLoader(filepath))
        
        meta.integration_time_us = int(temp_mesu.integration_time * 1000)
        meta.flags = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta.flags[key] = val
        meta.references = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            meta.references[key] = val
            
        self.metas.append(meta)


