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
from pycocotools.coco import COCO
from .Labels2TV import convert_COCO2TV
from enum import Enum

from .Metadata import Metadata

debug_enabled = True

class OutputFormat(Enum):
    """"
    Describes the output format that is return by the dataloader
    """
    Full = 0
    BoundingBox = 1
    SegmentationMask = 2
    CustomFilter = 99


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
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self._FILE_EXTENSION = ".npy"
        self.output_format = output_format
        self.output_lambda = output_lambda

        self.fileset_metadata = {}
        self.metadata_filepath = os.path.join(self.root, "metadata.yaml")
        if os.path.isfile(self.metadata_filepath):
            self.fileset_metadata = yaml.safe_load(open(self.metadata_filepath, "r"))
        else:
            self.metadata_filepath = ""

        self.data_types:set = set()
        self.provide_datatype:np.dtype = np.float32
        
        # Actual data collection
        self.paths = []
        self.cubes = []
        self.metas = []
        self.labels = []

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

        self.paths.append(filepath)
        self.cubes.append(self._NumpyLoader_(filepath))
        
            
        if self.metadata_filepath:
            meta = Metadata(filepath, self.fileset_metadata)
        else:
            meta = Metadata(filepath)

        temp_data = np.load(filepath)
        meta.shape = temp_data.shape
        meta.datatype = temp_data.dtype
        self.data_types.add(meta.datatype)

        self.metas.append(meta)

        labelpath = os.path.splitext(filepath)[0] + ".json"
        canvas_size = (meta.shape[0], meta.shape[1])
        if os.path.isfile(labelpath):
            coco = COCO(labelpath)
            l = convert_COCO2TV(coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0])), canvas_size)
        else:
            l = None
        self.labels.append()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.get_data(idx)
        meta = self.get_metadata(idx)
        label = self.get_labels(idx)
        return self._get_return_shape(data, meta, label)
    
    def _get_return_shape(self, data, metadata, labels):
        if self.output_format == OutputFormat.Full:
            return data, metadata, labels

        elif self.output_format == OutputFormat.BoundingBox:
            return data, [l['bbox'] for l in labels]
        
        elif self.output_format == OutputFormat.SegmentationMask:
            return data, [l['segmentation'] for l in labels]
        
        elif self.output_format == OutputFormat.CustomFilter and self.output_lambda is not None:
            return [self.output_lambda(d, m, l) for d, m, l in zip(data, metadata, labels)]
        
        else:
            raise NotImplementedError("Think about it.")

    def _apply_transform(self, d):
        return d if self.transforms is None else self.transforms(d)
    
    def merge(self, other_dataset):
        self.paths.extend(other_dataset.paths)
        self.cubes.extend(other_dataset.cubes)
        self.metas.extend(other_dataset.metas)
        self.labels.extend(other_dataset.labels)

    def set_datatype(self, dtype: np.dtype):
        if dtype in NumpyData.C_SUPPORTED_DTYPES:
            self.provide_datatype = dtype
        else:
            raise ValueError("Unsupported data type: {" + str(dtype.name) + " - use one of: " + str([d.name for d in NumpyData.C_SUPPORTED_DTYPES]))

    def random_split(self, train_percent, val_percent, test_percent) -> list[torch.utils.data.dataset.Subset]:
        gen = torch.torch.Generator().manual_seed(time.time_ns())
        return torch.utils.data.random_split(self, [train_percent, val_percent, test_percent], gen)
    
    def get_dataitems_datatypes(self) -> list:
        return list(self.data_types)

    def get_datatype(self):
        return self.provide_datatype
    
    def get_all_cubes(self):
        return [cube(self.provide_datatype) for cube in self.cubes]

    def get_data(self, idx):
        if isinstance(idx, int):
            return self._apply_transform(self.cubes[idx](self.provide_datatype))
        return [self._apply_transform(cube(self.provide_datatype)) for cube in self.cubes[idx]]

    def get_all_items(self):
        return self[:]
    
    def get_item(self, idx):
        return self[idx]

    def get_all_metadata(self):
        return self.metas

    def get_metadata(self, idx):
        return self.metas[idx]

    def get_all_labels(self):
        return [self.get_labels(idx) for idx in range(len(self.labels))]

    def get_labels(self, idx):
        if isinstance(idx, int):
            return self._apply_transform(self.labels[idx])
        return [self._apply_transform(l) for l in self.labels[idx]]

