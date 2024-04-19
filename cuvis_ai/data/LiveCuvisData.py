import os
import cuvis
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Union, Dict
import yaml
import torch
import time
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset

from .NumpyData import OutputFormat, NumpyData
from .Metadata import Metadata
from ..tv_transforms import WavelengthList


class LiveCuvisData(VisionDataset):
    """Representation of a live source of HSI data cubes in the form of a real or simulated Cubert camera.
    
    This class is a subclass of torchvisions VisionDataset which is a subclass
    of torch.utils.data.Dataset.
    Using the attribute :attr:`camera` you have full access to the :class:`cuvis.AcquisitionContext` wrapped in this class.
    LiveCuvisData acts as a generator.
    
    Args:
        path (str, optional): Path to either a SessionFile or the directory containing a factory file.
        simulate (bool, optional): Only applicable when a SessionFile is passed as :attr:`path`. Will simulate having the camera that recorded the file connected. Loops the contents of the SessionFile.
        transforms (callable, optional): A function/transforms that takes in an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        output_format (OutputFormat): Enum value that controls the output format of the dataset. See :class:`OutputFormat`
        output_lambda (callable, optional): Only used when :attr:`output_format` is set to `CustomFilter`. Before returning data, the full output of the dataset is passed through this function to allow for custom filtering.
        
    Note:
        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """
    
    
    
    def __init__(self, path:Optional[Union[str, Path]] = None, 
        simulate:Optional[bool] = False, 
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Optional[Callable] = None
    ):
        super().__init__(path, transforms, transform, target_transform)
        self.output_format = output_format
        self.output_lambda = output_lambda
        self.provide_datatype:np.dtype = np.float32
        self._clear()
        
        if path is not None:
            self.initialize(path, simulate)
    
    def _clear(self):
        self.initialized = False
        if self.initialized and self.continuous:
            self.continuous = False
        self._camera_recording = False
        self.camera = None
        self.simulate = False
        self.continuous = False
        
    def initialize(self, path:Union[str, Path], simulate:bool=False, force:bool=False) -> bool:
        """Try to connect to the camera of the factory file in :attr:`path` or load the Sessionfile :attr:`path`.
        
        Args:
            path (str, optional): Path to either a SessionFile or the directory containing a factory file.
            simulate (bool, optional): Only applicable when a SessionFile is passed as :attr:`path`. Will simulate having the camera that recorded the file connected. Loops the contents of the SessionFile.
            force (bool, optional): Force reload. 
        """
        if self.initialized:
            if force:
                self._clear()
            else:
                raise RuntimeError("Cannot initialize an already initialized LiveCuvisData. Use force=True if this was intended.")
        self.root = path
        self.simulate = simulate
        if os.path.splitext(path)[-1] == ".cu3s":
            sess = cuvis.SessionFile(path)
        if sess is not None:
            self.camera = cuvis.AcquisitionContext(sess, simulate=simulate)
        else:
            self.camera = cuvis.AcquisitionContext(cuvis.Calibration(path))
        
        timeout = 20.0
        while self.camera.state != cuvis.HardwareState.Online:
            time.sleep(0.2)
            timeout -= 0.2
            if timeout < 0:
                return False
        
        self.camera.operation_mode = cuvis.OperationMode.Software
        self.capture_timeout_ms = 1000
        self.initialized = True
    
    @property
    def continuous(self) -> bool:
        return self._camera_recording
    
    @continuous.setter
    def continuous(self, val:bool):
        """Start continuous recording"""
        if self._camera_recording != val:
            if not self._camera_recording:
                self.camera.operation_mode = cuvis.OperationMode.Internal
            self.camera.set_continuous(val)
            if self._camera_recording:
                self.camera.operation_mode = cuvis.OperationMode.Software
            self._camera_recording = val
    
    def _capture(self):
        mesu = None
        while mesu is None:
            try:
                if self.continuous:
                    mesu = self.camera.get_next_measurement(self.capture_timeout_ms)
                else:
                    mesu = self.camera.capture_at(self.capture_timeout_ms)
            except cuvis.General.SDKException:
                raise RuntimeError("Could not fetch a measurement from the device - Timeout")
        cube = mesu.data["cube"].array
        wl = mesu.data["cube"].wavelength
        if cube.dtype != self.provide_datatype:
            cube = cube.astype(self.provide_datatype)
        
        meta = Metadata(mesu.name)
        meta.integration_time_us = int(mesu.integration_time * 1000)
        meta.flags = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta.flags[key] = val
        meta.references = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            meta.references[key] = val
        meta.shape = cube.shape
        meta.wavelengths_nm = wl
        meta.processing_mode = mesu.processing_mode
        
        cube = tv_tensors.Image(cube)
        while len(cube.shape) < 4:
            cube = cube.unsqueeze(0)
        cube = (cube.to(memory_format=torch.channels_last))
        
        labels = {"wavelength": WavelengthList(wl)}
        return (cube, meta, labels)
    
    def __getitem__(self, idx):
        """Return next data element in the selected :attr:`OutputFormat`. idx is ignored.
        Default is `OutputFormat.Full`, tuple(cube, meta-data, labels)
        """
        return self.__next__()
    
    def __next__(self):
        """Return next data element in the selected :attr:`OutputFormat`.
        Default is `OutputFormat.Full`, tuple(cube, meta-data, labels)
        """
        cube, meta, labels = self._capture()
        # torchvision transforms don't yet respect the memory layout property of tensors. They assume NCHW while cubes are in NHWC
        cube = self._apply_transform(cube.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        labels = self._apply_transform(labels)
        return self._get_return_shape(cube, meta, labels)
    
    def set_datatype(self, dtype: np.dtype):
        """Specify a Numpy datatype to transform the cube into before returning it.
        Valid data types are:
        np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool_
        """
        if dtype in NumpyData.C_SUPPORTED_DTYPES:
            self.provide_datatype = dtype
        else:
            raise ValueError("Unsupported data type: {" + str(dtype.name) + " - use one of: " + str([d.name for d in NumpyData.C_SUPPORTED_DTYPES]))

    def _apply_transform(self, d):
        return d if self.transforms is None else self.transforms(d)
    
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

    def serialize(self, serial_dir: str):
        """Serialize the parameters of this dataset and store in 'serial_dir'."""
        if not self.initialized:
            print('Module not fully initialized, skipping output!')
            return
        
        blobname = F"{hash(self.transforms)}_dataset_transforms.zip"
        torch.save(self.transforms, os.path.join(serial_dir, blobname))
        data = {
            'type': type(self).__name__,
            'root_dir': self.root,
            'data_type': self.provide_datatype,
            'simulate': self.simulate,
            'transforms': blobname,
        }
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: Dict, filepath: str):
        """Load dumped parameters to recreate the dataset."""
        root = params["root_dir"]
        simulate = params["simulate"]
        self.provide_datatype = params["data_type"]
        self.transforms = torch.load(os.path.join(filepath, params["transforms"]))
        self.initialize(root, simulate)
