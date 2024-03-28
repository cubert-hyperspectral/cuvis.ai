import os
import glob
import yaml
import json
import time
from typing import Optional, Callable, Dict
import numpy as np
import torch
import torchvision
from torchvision.datasets import VisionDataset 
from torchvision import tv_tensors
from pycocotools.coco import COCO
from .Labels2TV import convert_COCO2TV
from enum import Enum
from ..transforms import Bandpass, MultiBandpass, WavelengthList

from .Metadata import Metadata

debug_enabled = True

C_SUPPORTED_DTYPES = (np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool_)


class OutputFormat(Enum):
    """"
    Describes the output format that is returned by NumpyData and CuvisData.
    Possible values and their respective return formats:
        - Full: (cube, meta.data, label data)
        - BoundingBox: (cube, torchvision BoundingBoxes)
        - SegmentationMask: (cube, torchvision Mask)
        - CustomFilter: output_lambda((cube, meta.data, label data))
    """
    Full = 0
    BoundingBox = 1
    SegmentationMask = 2
    CustomFilter = 99


class NumpyData(VisionDataset):
    """Representation for a set of data cubes, their meta-data and labels.
    
    This class is a subclass of torchvisions VisionDataset which is a subclass
    of torch.utils.data.Dataset.
    This class can be used anywhere these classes can be used, such as initializing
    a pytorch dataloader.
    
    Upon creation, the path set via :attr:`root` is scanned recursively for any compatible data.
    Any with the .npy extension are incorporated into the dataset.
    Metadata that is valid for the entire dataset is expected to be in the file `root`/metadata.yaml.
    See :class:`Metadata` for an explanation of the data format.
    Labels for any data file are expected in a file of the same name but with the .json extension in COCO label format.
    
    Args:
        root (str, optional): The absolute or relative path to the directory containing the HSI data.
        transforms (callable, optional): A function/transforms that takes in an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        output_format (OutputFormat): Enum value that controls the output format of the dataset. See :class:`OutputFormat`
        output_lambda (callable, optional): Only used when :attr:`output_format` is set to `CustomFilter`. Before returning data, the full output of the dataset is passed through this function to allow for custom filtering.
        
    Note:
        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    
    Note:
        If :attr:`root` is not passed in the constructor, the :py:meth:`~NumpyData.initialize` or :py:meth:`~NumpyData.load` method has to be called with a root path before the dataset can be used.
    """
    
    class _NumpyLoader_:
        def __init__(self, path):
            self.path = path
        def __call__(self, to_dtype:np.dtype):
            cube = np.load(self.path)
            if cube.dtype != to_dtype:
                cube = cube.astype(to_dtype)
            cube = tv_tensors.Image(cube)
            while len(cube.shape) < 4:
                cube = cube.unsqueeze(0)
            return cube.to(memory_format=torch.channels_last)

    def __init__(self, root: Optional[str] = None, 
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.output_format = output_format
        self.output_lambda = output_lambda
        
        self._FILE_EXTENSION = ".npy"
        self.provide_datatype:np.dtype = np.float32
        
        self._clear()

        self.bandpass = [t for t in transforms.transforms if isinstance(t, Bandpass) or isinstance(t, MultiBandpass)]
        if len(self.bandpass) > 0:
            self.bandpass = torchvision.transforms.v2.Compose(self.bandpass)
        else:
            self.bandpass = None
        
        if root is not None:
            self.initialize(root)


    def initialize(self, root:str, force:bool = False):
        """ Initialize the dataset by scanning the provided directory for data.
        Initialize will be called by the constructor if a root path is provided or by the load method.
        
        Args:
            root: Path of the directory containing the data this dataset will represent.
            force: If True, the dataset will clear all currently held data and re-initialize with the provided root path.
        """
        if self.initialized:
            if force:
                self._clear()
            else:
                raise RuntimeError("Cannot initialize an already initialized dataset. Use force=True if this was intended.")
        self.root = root

        self.metadata_filepath = os.path.join(self.root, "metadata.yaml")
        if os.path.isfile(self.metadata_filepath):
            self.fileset_metadata = yaml.safe_load(open(self.metadata_filepath, "r"))
        else:
            self.metadata_filepath = ""

        # Actual data collection
        self._load_directory(self.root)
        self.initialized = True
    
    def _clear(self):
        self.root = None
        self.paths = []
        self.cubes = []
        self.metas = []
        self.labels = []
        self.fileset_metadata = {}
        self.data_types:set = set()
        self.initialized = False
        
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
            anns = coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0]))
            try:
                anns["wavelength"] = coco.imgs[0]["wavelength"]
            except KeyError:
                pass
            
            l = convert_COCO2TV(anns, canvas_size)
        else:
            l = None
        self.labels.append()

    def __len__(self):
        """The number of data elements this data set holds."""
        return len(self.cubes)

    def __getitem__(self, idx):
        """Return data element number 'idx' in the selected :attr:`OutputFormat`.
        Default is `OutputFormat.Full`, tuple(cube, meta-data, labels):
        """
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
        """Merge another NumpyData dataset into this dataset."""
        if type(self) is type(other_dataset):
            self.paths.extend(other_dataset.paths)
            self.cubes.extend(other_dataset.cubes)
            self.metas.extend(other_dataset.metas)
            self.labels.extend(other_dataset.labels)
        else:
            raise TypeError(F"Cannot merge NumpyData with an object of type {type(other_dataset).__name__}")

    def set_datatype(self, dtype: np.dtype):
        """Specify a Numpy datatype to transform the cube into before returning it.
        Valid data types are:
        np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool_
        """
        if dtype in NumpyData.C_SUPPORTED_DTYPES:
            self.provide_datatype = dtype
        else:
            raise ValueError("Unsupported data type: {" + str(dtype.name) + " - use one of: " + str([d.name for d in NumpyData.C_SUPPORTED_DTYPES]))

    def random_split(self, train_percent, val_percent, test_percent) -> list[torch.utils.data.dataset.Subset]:
        """Generate three datasets with randomly chosen data from this dataset.
        Args:
            train_percent (float): How much of the data to put into the training dataset.
            val_percent (float): How much of the data to put into the validation dataset.
            test_percent (float): How much of the data to put into the testing dataset.
        
        Returns:
            tuple: (train, val, test) datasets
        """
        gen = torch.torch.Generator().manual_seed(time.time_ns())
        return torch.utils.data.random_split(self, [train_percent, val_percent, test_percent], gen)
    
    def get_dataitems_datatypes(self) -> list:
        """Get a list with all datatypes detected when scanning the root folder."""
        return list(self.data_types)

    def get_datatype(self):
        """Get the current datatype set that all data will be converted into before return."""
        return self.provide_datatype
    
    def get_all_cubes(self):
        """Get a list of all cubes in this dataset.
        Note:
            Not recommended for large sets. All data will be read into RAM!
        """
        return [cube(self.provide_datatype) for cube in self.cubes]

    def get_data(self, idx):
        """Get the cube at 'idx'."""
        # torchvision transforms don't yet respect the memory layout property of tensors. They assume NCHW while cubes are in NHWC
        if isinstance(idx, int):
            return self._apply_transform(self.cubes[idx](self.provide_datatype).permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        return [self._apply_transform(cube(self.provide_datatype).permute([0, 3, 1, 2])).permute([0, 2, 3, 1]) for cube in self.cubes[idx]]

    def get_all_items(self):
        """Get all items of this dataset in the selected :attr:`output_format`.
        Note:
            Not recommended for large sets. All data will be read into RAM!"""
        return self[:]
    
    def get_item(self, idx):
        """Get item at 'idx' of this dataset in the selected :attr:`output_format`."""
        return self[idx]

    def get_all_metadata(self):
        """Get the meta-data for every cube in this dataset."""
        return self.metas

    def get_metadata(self, idx):
        """Get the meta-data for the cube at 'idx' in this dataset."""
        return self.metas[idx]

    def get_all_labels(self):
        """Get the labels for every cube in this dataset."""
        return [self.get_labels(idx) for idx in range(len(self.labels))]

    def get_labels(self, idx):
        """Get the labels for the cube at 'idx' in this dataset."""
        if isinstance(idx, int):
            ret = self._apply_transform(self.labels[idx])
            if self.bandpass is not None:
                try:
                    ret["wavelength"] = self.bandpass(WavelengthList(ret["wavelength"])).numpy()
                except KeyError:
                    pass
            return ret
        ret = [self._apply_transform(l) for l in self.labels[idx]]
        if self.bandpass is not None:
            for r in ret:
                try:
                    r["wavelength"] = self.bandpass(WavelengthList(r["wavelength"])).to_numpy()
                except KeyError:
                    pass
        return ret

    def serialize(self, serial_dir: str):
        """Serialize the parameters of this dataset and store in 'serial_dir'."""
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        
        blobname = F"{hash(self.transforms)}_dataset_transforms.zip"
        torch.save(self.transforms, os.path.join(serial_dir, blobname))
        data = {
            'dataset': type(self).__name__,
            'root_dir': self.root,
            'data_type': self.provide_datatype,
            'transforms': blobname,
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: Dict, filepath: str):
        """Load dumped parameters to recreate the dataset."""
        root = params["root_dir"]
        self.provide_datatype = params["data_type"]
        self.transforms = torch.load(os.path.join(filepath, params["transforms"]))
        self.initialize(root)
