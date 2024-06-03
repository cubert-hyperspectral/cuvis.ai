import os
import cuvis
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Union, Dict
import yaml
import torch
import time
import uuid
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset

from cuvis.General import SDKException

from .NumpyData import OutputFormat, NumpyData
from .Metadata import Metadata
from ..tv_transforms import WavelengthList


class LiveCuvisData(VisionDataset):
    """Representation of a live source of HSI data cubes in the form of a real or simulated Cubert camera.
    
    This class is a subclass of torchvisions VisionDataset which is a subclass
    of torch.utils.data.Dataset.
    Using the attribute :attr:`camera` you have full access to the :class:`cuvis.AcquisitionContext` wrapped in this class.
    Additionally, using the attribute :attr:`processing_context` you have full access to the :class:`cuvis.ProcessingContext` used to process the raw data from the (simulated) camera.
    LiveCuvisData acts as a generator.
    
    Parameters
    ----------
    path : str, optional
        Path to either a SessionFile or the directory containing a factory file.
    transforms : callable, optional
        A function/transforms that takes in an image and a label and returns the transformed versions of both.
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version. E.g, transforms.RandomCrop
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    output_format : OutputFormat
        Enum value that controls the output format of the dataset. See :class:`OutputFormat`
    output_lambda : callable, optional
        Only used when :attr:`output_format` is set to `CustomFilter`. Before returning data, the full output of the dataset is passed through this function to allow for custom filtering.
        
    Notes
    -----
    :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    
    If :attr:`path` is not passed in the constructor, the :py:meth:`~LiveCuvisData.initialize` or :py:meth:`~LiveCuvisData.load` method has to be called with a path before the object can be used.
    """
    
    _cuvis_refmap = {
        "dark_ref": cuvis.ReferenceType.Dark,
        "whitedark_ref": cuvis.ReferenceType.WhiteDark,
        "white_ref": cuvis.ReferenceType.White,
        "distancecalib_ref": cuvis.ReferenceType.Distance,
        "spradcalib_ref": cuvis.ReferenceType.SpRad,
    }
    _cuvis_non_cube_references = (cuvis.ReferenceType.Distance, cuvis.ReferenceType.SpRad)
    
    def __init__(self, path:Optional[Union[str, Path]] = None, 
        simulate:Optional[bool] = False, 
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        output_format: OutputFormat = OutputFormat.Full,
        output_lambda: Optional[Callable] = None
    ):
        super().__init__(path, transforms, transform, target_transform)
        self.id = F"{self.__class__.__name__}-{str(uuid.uuid4())}"
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
        self.processing_context = None
        self.sess = None
        self._refcache = {}
        
    def initialize(self, path:Union[str, Path], simulate:bool=False, force:bool=False, timeout_s:int=20) -> bool:
        """Try to connect to the camera of the factory file in :attr:`path` or load the Sessionfile :attr:`path`.
        
        Parameters
        ----------
        path : str, optional
            Path to either a SessionFile or the directory containing a factory file.
        simulate : bool, optional
            Only applicable when a SessionFile is passed as :attr:`path`. Will simulate having the camera that recorded the file connected. Loops the contents of the SessionFile.
        force : bool, optional
            Force reload.
        timeout_s : int, optional
            Duration in seconds to wait for the device to establish a connection before returning. Default is 20 seconds
        
        Returns
        -------
        False if the device does not come online before the timeout specified by :attr:`timeout_s`, True otherwise.
        """
        if self.initialized:
            if force:
                self._clear()
            else:
                raise RuntimeError("Cannot initialize an already initialized LiveCuvisData. Use force=True if this was intended.")
        self.root = path
        self.simulate = simulate
        
        if os.path.splitext(path)[-1] == ".cu3s":
            self.sess = cuvis.SessionFile(path)
        if self.sess is not None:
            self.camera = cuvis.AcquisitionContext(self.sess, simulate=simulate)
            self.processing_context = cuvis.ProcessingContext(self.sess)
        else:
            calib = cuvis.Calibration(path)
            self.camera = cuvis.AcquisitionContext(calib)
            self.processing_context = cuvis.ProcessingContext(calib)
        
        self.processing_context.processing_mode = cuvis.ProcessingMode.Raw
        
        timeout = timeout_s
        while self.camera.state != cuvis.HardwareState.Online:
            time.sleep(0.2)
            timeout -= 0.2
            if timeout < 0:
                return False
        
        self.camera.operation_mode = cuvis.OperationMode.Software
        self.capture_timeout_ms = 1000
        self.initialized = True
        return True
    
    @property
    def continuous(self) -> bool:
        return self._camera_recording
    
    @continuous.setter
    def continuous(self, val:bool):
        """Start or stop continuous recording.
        Parameters
        ----------
        val : bool
            Pass `True` to start recording. Pass `False` to stop recording.
        """
        if self._camera_recording != val:
            if not self._camera_recording:
                self.camera.operation_mode = cuvis.OperationMode.Internal
            self.camera.set_continuous(val)
            if self._camera_recording:
                self.camera.operation_mode = cuvis.OperationMode.Software
            self._camera_recording = val

    @property
    def processing_mode(self) -> cuvis.ProcessingMode:
        return self.processing_context.processing_mode
    
    @processing_mode.setter
    def processing_mode(self, val:cuvis.ProcessingMode):
        """Set the processing mode for preprocessing of the data.
        Parameters
        ----------
        val : cuvis.ProcessingMode
            Valid modes are `Raw`, `DarkSubtract`, `Reflectance` and `SpectralRadiance`.
            All modes except for `Raw` require a dark reference to be set using :meth:`set_reference` or :meth:`record_dark`.
            The mode `Reflectance` additionally requires a white reference to be set using :meth:`set_reference` or :meth:`record_white`.
        """
        self.processing_context.processing_mode = val
    
    def _fetch_mesu(self) -> cuvis.Measurement:
        mesu = None
        while mesu is None:
            try:
                if self.continuous:
                    mesu = self.camera.get_next_measurement(self.capture_timeout_ms)
                else:
                    mesu = self.camera.capture_at(self.capture_timeout_ms)
            except SDKException:
                raise RuntimeError("Could not fetch a measurement from the device - Timeout")
        return mesu
    
    def _fetch_averaged_mesu(self, averaging_count:int) -> cuvis.Measurement:
        avg_prev = self.camera.average
        timeout_prev = self.capture_timeout_ms

        self.camera.average = averaging_count
        self.capture_timeout_ms = timeout_prev * averaging_count + 1000
        
        mesu = self._fetch_mesu()
        
        self.camera.average = avg_prev
        self.capture_timeout_ms = timeout_prev

        return mesu

    def set_reference(self, mesu: cuvis.Measurement, reftype: cuvis.ReferenceType):
        """Associate a reference measurement with this acquisition session.
        Parameters
        ----------
        mesu : cuvis.Measurement
            The existing measurement to use as the reference.
        reftype : cuvis.ReferenceType
            The type of reference to use this measurement as.
        """
        #print(F"Adding reference mesu: [{reftype.name} -> {mesu.name} ({list(mesu.data.keys())})]")
        self.processing_context.set_reference(mesu, reftype)
        if reftype not in self._cuvis_non_cube_references:
            self._refcache[reftype.name] = mesu

    def record_dark(self, averaging_count:int = 5):
        """Record a dark reference measurement using this acquisition session.
        Parameters
        ----------
        averaging_count : int
            The number of measurements to record and average into one measurement. This can improve the quality of reference measurement by reducing the impact of noise. Default is 5.
        """
        avg = max(1, averaging_count)
        self.set_reference(self._fetch_averaged_mesu(avg), cuvis.ReferenceType.Dark)
    
    def record_white(self, averaging_count:int = 5):
        """Record a white reference measurement using this acquisition session.
        Parameters
        ----------
        averaging_count : int
            The number of measurements to record and average into one measurement. This can improve the quality of reference measurement by reducing the impact of noise. Default is 5.
        """
        avg = max(1, averaging_count)
        self.set_reference(self._fetch_averaged_mesu(avg), cuvis.ReferenceType.White)
    
    def record_distance(self):
        """Record a distance reference measurement using this acquisition session.
        The camera should be pointed at a scene with high contrast at the distance that you want to observe.
        Cuvis will attempt to reduce the parallax effect present in the HSI camera by live-tuning on the captured image.
        """
        self.set_reference(self._fetch_mesu(), cuvis.ReferenceType.Distance)
    
    @staticmethod
    def _mesu_to_tensor(mesu:cuvis.Measurement, astype:np.dtype) -> tv_tensors.Image:
        try:
            cube = mesu.data["cube"].array
        except KeyError:
            print(list(mesu.data.keys()))
            raise ValueError(F"No HSI cube in measurement {mesu.name}!")
        
        if cube.dtype != astype:
            cube = cube.astype(astype)
        cube = tv_tensors.Image(cube)
        while len(cube.shape) < 4:
            cube = cube.unsqueeze(0)
        cube = (cube.to(memory_format=torch.channels_last))
        return cube
    
    def _capture(self):
        mesu = self._fetch_mesu()
        
        # Load references from session file
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            reftype = LiveCuvisData._cuvis_refmap[key]
            hasref = self.processing_context.has_reference(reftype)
            if (self.sess is not None) and (not hasref):
                
                refcount = 0
                while True:
                    try:
                        refmesu = self.sess.get_reference(refcount, reftype)
                    except SDKException:
                        refmesu = None
                        break
                    if refmesu is not None and refmesu.name in val:
                        break
                    refcount += 1
                if refmesu is not None:
                    self.set_reference(refmesu, reftype)
        reprocessed = False
        if mesu.processing_mode != self.processing_mode:
            mesu = self.processing_context.apply(mesu)
            reprocessed = True
            mesu.refresh()
        
        wl = mesu.data["cube"].wavelength
        
        cube = self._mesu_to_tensor(mesu, self.provide_datatype)
        
        meta = Metadata(mesu.name)
        meta.integration_time_us = int(mesu.integration_time * 1000)
        meta.shape = cube.shape
        meta.wavelengths_nm = wl
        meta.processing_mode = mesu.processing_mode
        
        meta.flags = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta.flags[key] = val
        
        meta.references = {}
        for reftype, refmesu in self._refcache.items():
            meta.references[reftype] = self._mesu_to_tensor(refmesu, self.provide_datatype)
        
        labels = {"wavelength": WavelengthList(wl)}
        
        return (cube, labels, meta)
    
    def __getitem__(self, idx:int):
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
        return self._get_return_shape(cube, [labels], [meta])
    
    def set_datatype(self, dtype: np.dtype):
        """Specify a Numpy datatype to transform the cube into before returning it.
        Valid data types are:
        np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool_
        """
        if dtype in NumpyData.C_SUPPORTED_DTYPES:
            self.provide_datatype = dtype
        else:
            raise ValueError("Unsupported data type: {" + str(dtype.name) + " - use one of: " + str([d.name for d in NumpyData.C_SUPPORTED_DTYPES]))

    def get_references(self) -> Dict[cuvis.ReferenceType, cuvis.Measurement]:
        return self._refcache

    def _get_return_shape(self, data, labels, metadata):
        if self.output_format == OutputFormat.Full:
            return (data, labels, metadata)

        elif self.output_format == OutputFormat.BoundingBox:
            return (data, [l['bbox'] for l in labels])
        
        elif self.output_format == OutputFormat.SegmentationMask:
            return (data, [l['segmentation'] for l in labels])
        
        elif self.output_format == OutputFormat.CustomFilter and self.output_lambda is not None:
            return self.output_lambda(data, labels, metadata)
        
        else:
            raise NotImplementedError("Think about it.")

    def _apply_transform(self, d):
        def unTensorify(source):
            if isinstance(source, dict):
                for k, v in source.items():
                    if isinstance(v, torch.Tensor):
                        source[k] = v.numpy()
                    elif isinstance(v, dict):
                        unTensorify(source[k])
            return source
        return d if self.transforms is None else unTensorify(self.transforms(d))

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
            'processing_mode': self.processing_contextessing_mode.value,
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
        self.processing_contextessing_mode = cuvis.ProcessingMode(params["processing_mode"])
