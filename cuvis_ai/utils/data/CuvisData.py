import os
os.environ["CUVIS"] = "/usr/lib/cuvis/"
import cuvis
import numpy as np
import glob
import yaml
import json
import copy
import torch
from torch.utils.data import Dataset
from imantics import Dataset as Labelparser

from .Metadata import Metadata
from .NumpyData import NumpyData

debug_enabled = True

class CuvisData(NumpyData):

    class _SessionCubeLoader:
        def __init__(self, path, idx):
            self.path = path
            self.idx = idx
        def __call__(self, to_dtype:np.dtype):
            return torch.as_tensor(cuvis.SessionFile(self.path).get_measurement(self.idx).Data["cube"].array.astype(to_dtype))
    
    class _LegacyCubeLoader:
        def __init__(self, path):
            self.path = path
        def __call__(self, to_dtype:np.dtype):
            return torch.as_tensor(cuvis.Measurement.load(self.path).Data["cube"].array.astype(to_dtype))
    
    def __init__(self, data_directory_path: str):
        self._FILE_EXTENSION_SESSION = ".cu3s"
        self._FILE_EXTENSION_LEGACY = ".cu3"
        super().__init__(data_directory_path)
        

    def _load_directory(self, dir_path:str):
        if debug_enabled:
            print("Reading from directory:", dir_path)
        fileset_session = glob.glob(os.path.join(self.path, '**/*' + self._FILE_EXTENSION_SESSION), recursive=True)
        
        fileset_legacy = glob.glob(os.path.join(self.path, '**/*' + self._FILE_EXTENSION_LEGACY), recursive=True)
        
        for cur_path in fileset_session:
            self._load_session_file(cur_path)
        for cur_path in fileset_legacy:
            self._load_legacy_file(cur_path)
            
    def _load_session_file(self, filepath: str):
        print("Found file:", filepath)
        path, _ = os.path.splitext(filepath)
        labelpath = path + ".json"

        crt_session = cuvis.SessionFile(filepath)

        cube_count = len(crt_session)
        print("Session file has", cube_count, "cubes")

        lp:Labelparser = None
        
        if os.path.isfile(labelpath):
            lp = Labelparser(filepath)
            with open(labelpath, "r") as file:
                lp.from_coco(json.load(file))

        if self.metadata_filepath:
            sess_meta = Metadata(filepath, self.fileset_metadata)
        else:
            sess_meta = Metadata(filepath)
        
        temp_mesu = crt_session.get_measurement(0)
        sess_meta.shape = (temp_mesu.data["cube"].width, temp_mesu.data["cube"].height, temp_mesu.data["cube"].channels)
        sess_meta.wavelengths_nm = temp_mesu.data["cube"].wavelength
        sess_meta.framerate = crt_session.fps
        
        for idx in range(cube_count):
            cube_path = F"{filepath}:{idx}"
            self.data[cube_path] = {}
            self.data[cube_path]["data"] = self._SessionCubeLoader(filepath, idx)
            
            mesu = crt_session.get_measurement(idx)
            
            meta:Metadata = copy.deepcopy(sess_meta)
            meta.integration_time_us = int(mesu.integration_time * 1000)
            meta.flags = {}
            for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
                meta.flags[key] = val
            meta.references = {}
            for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
                meta.references[key] = val
            self.data[cube_path]["meta"] = meta

            if lp is not None:
                mesu_lp = Labelparser(cube_path)
                mesu_lp.name = cube_path
                
                mesu_lp.images[idx] = mesu_lp.images[idx]
                for k, v in mesu_lp.images[idx].annotations.items():
                    mesu_lp.annotations[k] = v
                self.data[cube_path]["labels"] = mesu_lp
            else:
                self.data[cube_path]["labels"] = None

    def _load_legacy_file(self, filepath:str):
        print("Found file:", filepath)
        path, _ = os.path.splitext(filepath)
        labelpath = path + ".json"
        
        lp:Labelparser = None
        
        if os.path.isfile(labelpath):
            self.data[path]["labels"] = Labelparser(filepath)
            with open(labelpath, "r") as file:
                self.data[path]["labels"].from_coco(json.load(file))
        else:
            self.data[filepath]["labels"] = None
                
        if self.metadata_filepath:
            meta = Metadata(filepath, self.fileset_metadata)
        else:
            meta = Metadata(filepath)
        
        temp_mesu = cuvis.Measurement(filepath)
        meta.shape = (temp_mesu.data["cube"].width, temp_mesu.data["cube"].height, temp_mesu.data["cube"].channels)
        meta.wavelengths_nm = temp_mesu.data["cube"].wavelength
        
        self.data[cube_path] = {}
        self.data[cube_path]["data"] = self._LegacyCubeLoader(filepath)
        
        meta.integration_time_us = int(mesu.integration_time * 1000)
        meta.flags = {}
        for key, val in [(key, temp_mesu.data[key]) for key in temp_mesu.data.keys() if "Flag_" in key]:
            meta.flags[key] = val
        meta.references = {}
        for key, val in [(key, temp_mesu.data[key]) for key in temp_mesu.data.keys() if "_ref" in key]:
            meta.references[key] = val
        self.data[filepath]["meta"] = meta


