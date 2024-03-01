import os
import glob
import yaml
import json
import time
from typing import Optional, Callable
import numpy as np
import torch
from torchvision.datasets import VisionDataset 
from torchvision import tv_tensors
from imantics import Dataset as Labelparser
from pycocotools.coco import COCO

from .Metadata import Metadata

debug_enabled = True

class NumpyData(VisionDataset):

    C_SUPPORTED_DTYPES = (np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool_)

    class _NumpyLoader_:
        def __init__(self, path):
            self.path = path
        def __call__(self, to_dtype:np.dtype):
            return tv_tensors.Image(np.load(self.path).astype(to_dtype))
    
    def __init__(self, root: str, 
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self._FILE_EXTENSION = ".npy"
        
        self.fileset_metadata = {}
        self.metadata_filepath = os.path.join(self.root, "metadata.yaml")
        if os.path.isfile(self.metadata_filepath):
            self.fileset_metadata = yaml.safe_load(open(self.metadata_filepath, "r"))
        else:
            self.metadata_filepath = ""

        self.data_types:set = set()
        self.provide_datatype:np.dtype = np.float32
        
        self.data_map:dict = {}
        self._load_directory(self.root)

    
    def _load_directory(self):
        if debug_enabled:
            print("Reading from directory:", self.root)
        fileset = glob.glob(os.path.join(self.root, '**/*' + self._FILE_EXTENSION), recursive=True)

        for cur_path in fileset:
            self._load_file(cur_path)
    
    def _load_file(self, filepath: str):
        if debug_enabled:
            print("Found file:", filepath)
        path, _ = os.path.splitext(filepath)
        labelpath = path + ".json"
        self.data_map[path] = {}
        self.data_map[path]["data"] = self._NumpyLoader_(filepath)
        
        if os.path.isfile(labelpath):
#            lp = Labelparser(filepath)
#            with open(labelpath, "r") as file:
#                lp.from_coco(json.load(file))
            coco = COCO(labelpath)
            self.data_map[path]["labels"] = coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0]))
        else:
            self.data_map[path]["labels"] = None
            
        if self.metadata_filepath:
            meta = Metadata(filepath, self.fileset_metadata)
        else:
            meta = Metadata(filepath)

        temp_data = np.load(filepath)
        meta.shape = temp_data.shape
        meta.datatype = temp_data.dtype
        self.data_types.add(meta.datatype)
        self.data_map[path]["meta"] = meta
        

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx: int):
        data = list(self.data_map.values())[idx]["data"](self.provide_datatype)
        labels = self.get_labels(idx)
        if self.transforms is not None:
            if labels is not None:
                data, labels = self.transforms(data, labels)
            else:
                data = self.transforms(data)
        return data, self.get_metadata(idx), labels
    
    def update(self, other_dataset):
        self.data_map.update(other_dataset.get_all_data())

    def set_datatype(self, dtype: np.dtype):
        if dtype in C_SUPPORTED_DTYPES:
            self.provide_datatype = dtype
        else:
            raise ValueError("Unsupported data type: {" + str(dtype.name) + " - use one of: " + str([d.name for d in C_SUPPORTED_DTYPES]))

    def random_split(self, train_percent, val_percent, test_percent) -> list[torch.utils.data.dataset.Subset]:
        gen = torch.torch.Generator().manual_seed(time.time_ns())
        return torch.utils.data.random_split(self, [train_percent, val_percent, test_percent], gen)
    
    def get_dataitems_datatypes(self) -> list:
        return list(self.data_types)

    def get_datatype(self):
        return self.provide_datatype
    
    def get_all_data(self):
        return self.data_map

    def get_data(self, idx:int):
        return list(self.data_map.items())[idx]

    def get_all_items(self):
        return [i["data"](self.provide_datatype) for i in self.data_map.values()]
    
    def get_item(self, idx:int):
        return self.__getitem__(idx)

    def get_all_metadata(self):
        return [i["meta"] for i in self.data_map.values()]

    def get_metadata(self, idx:int):
        return list(self.data_map.values())[idx]["meta"]

    def get_all_labels(self):
        def _get_labels(v):
            try:
                return v["labels"]
            except KeyError:
                return None
        return [_get_labels(i) for i in self.data_map.values()]

    def get_labels(self, idx:int):
        try:
            return list(self.data_map.values())[idx]["labels"]
        except KeyError:
            return None

