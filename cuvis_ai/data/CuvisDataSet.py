import os
import cuvis
import numpy as np
import glob
import copy
from typing import Optional, Callable, Dict
import torch
import uuid
from torchvision import tv_tensors
from pycocotools.coco import COCO

from cuvis.General import SDKException

from .Labels2TV import convert_COCO2TV
from .MetadataUtils import metadataInit
from .NumpyDataSet import NumpyDataSet
from .OutputFormat import OutputFormat
from ..tv_transforms import WavelengthList

debug_enabled = True


class CuvisDataSet(NumpyDataSet):
    """Representation for a set of Cuvis data cubes, their meta-data and labels.

    See :class:`NumpyData` for more details.


    Parameters
    ----------
    root : str, optional
        The absolute or relative path to the directory containing the HSI data.
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

    If :attr:`root` is not passed in the constructor, the :py:meth:`~CuvisDataSet.initialize` or :py:meth:`~CuvisDataSet.load` method has to be called with a root path before the dataset can be used.
    """
    _cuvis_non_cube_references = (
        cuvis.ReferenceType.Distance, cuvis.ReferenceType.SpRad)

    class _SessionCubeLoader_:
        def __init__(self, path, idx):
            self.path = path
            self.idx = idx

        def __call__(self, to_dtype: np.dtype):
            cube = cuvis.SessionFile(self.path).get_measurement(
                self.idx).data["cube"].array
            if cube.dtype != to_dtype:
                cube = cube.astype(to_dtype)
            cube = tv_tensors.Image(cube)
            while len(cube.shape) < 4:
                cube = cube.unsqueeze(0)
            return cube.to(memory_format=torch.channels_last)

    class _SessionReferenceLoader_:
        def __init__(self, path, reftype):
            self.path = path
            self.reftype = reftype

        def __call__(self, to_dtype: np.dtype):
            cube = cuvis.SessionFile(self.path).get_reference(
                0, self.reftype).data["cube"].array
            if cube.dtype != to_dtype:
                cube = cube.astype(to_dtype)
            cube = tv_tensors.Image(cube)
            while len(cube.shape) < 4:
                cube = cube.unsqueeze(0)
            return cube.to(memory_format=torch.channels_last)

    class _LegacyCubeLoader_:
        def __init__(self, path):
            self.path = path

        def __call__(self, to_dtype: np.dtype):
            cube = cuvis.Measurement.load(self.path).data["cube"].array
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
        self._FILE_EXTENSION_SESSION = ".cu3s"
        self._FILE_EXTENSION_LEGACY = ".cu3"
        super().__init__(root, transforms=transforms, transform=transform,
                         target_transform=target_transform, output_format=output_format, output_lambda=output_lambda)

    def _load_directory(self, dir_path: str):
        if debug_enabled:
            print("Reading from directory:", dir_path)
        fileset_session = glob.glob(os.path.join(
            self.root, '**/*' + self._FILE_EXTENSION_SESSION), recursive=True)

        fileset_legacy = glob.glob(os.path.join(
            self.root, '**/*' + self._FILE_EXTENSION_LEGACY), recursive=True)

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

        sess_meta = metadataInit(filepath, self.fileset_metadata)

        temp_mesu = crt_session.get_measurement(0)
        sess_meta["shape"] = (temp_mesu.data["cube"].width,
                              temp_mesu.data["cube"].height, temp_mesu.data["cube"].channels)
        canvas_size = (sess_meta["shape"][0], sess_meta["shape"][1])
        sess_meta["wavelengths_nm"] = WavelengthList(
            temp_mesu.data["cube"].wavelength)
        try:
            sess_meta["framerate"] = crt_session.fps
        except SDKException:
            pass

        try:
            sess_meta["references"]
        except KeyError:
            sess_meta["references"] = {}

        for reftype in cuvis.ReferenceType:
            try:
                sess_meta["references"][reftype]
            except KeyError:
                try:
                    refmesu = crt_session.get_reference(0, reftype)
                except SDKException:
                    refmesu = None
                if refmesu is not None and reftype not in self._cuvis_non_cube_references:
                    sess_meta["references"][reftype.name] = self._SessionReferenceLoader_(
                        filepath, reftype)

        coco = None
        if os.path.isfile(labelpath):
            coco = COCO(labelpath)
            ids = list(sorted(coco.imgs.keys()))

        for idx in range(cube_count):
            cube_path = F"{filepath}:{idx}"
            self.paths.append(cube_path)
            self.cubes.append(self._SessionCubeLoader_(filepath, idx))

            meta = copy.deepcopy(sess_meta)
            # TODO: Add a way to SDK where only meta data is loaded or make the cube lazy-loadable
            # mesu = crt_session.get_measurement(idx)
            # meta["integration_time_us"] = int(mesu.integration_time * 1000)
            # meta["flags"] = {}
            # for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            #    meta["flags"][key] = val
            # meta["references"] = {}
            # for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            #    meta["references"][key] = val
            for _, v in meta["references"].items():
                if isinstance(v, str):
                    if os.path.splitext(v)[-1] == ".cu3s":
                        v = self._SessionCubeLoader_(v)
                    elif os.path.splitext(v)[-1] == ".cu3":
                        v = self._LegacyCubeLoader_(v)

            self.metas.append(meta)

            l = {}
            if coco is not None:
                anns = coco.loadAnns(coco.getAnnIds(ids[idx]))[0]
                try:
                    anns["wavelength"] = coco.imgs[ids[idx]]["wavelength"]
                except KeyError:
                    pass
                l = convert_COCO2TV(anns, canvas_size)
            self.labels.append(l)

    def _load_legacy_file(self, filepath: str):
        if debug_enabled:
            print("Found file:", filepath)
        self.paths.append(filepath)
        labelpath = os.path.splitext(filepath)[0] + ".json"

        if self.metadata_filepath:
            meta = Metadata(filepath, self.fileset_metadata)
        else:
            meta = Metadata(filepath)

        mesu = cuvis.Measurement(filepath)
        meta["shape"] = (mesu.data["cube"].width,
                         mesu.data["cube"].height, mesu.data["cube"].channels)
        meta["wavelengths_nm"] = WavelengthList(mesu.data["cube"].wavelength)

        l = None
        canvas_size = (meta["shape"][0], meta["shape"][1])
        if os.path.isfile(labelpath):
            coco = COCO(labelpath)
            anns = coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0]))[0]
            try:
                anns["wavelength"] = coco.imgs[0]["wavelength"]
            except KeyError:
                pass

            l = convert_COCO2TV(anns, canvas_size)
        self.labels.append(l)

        self.cubes.append(self._LegacyCubeLoader_(filepath))

        meta["integration_time_us"] = int(mesu.integration_time * 1000)
        meta["flags"] = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta["flags"][key] = val
        meta["references"] = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            meta["references"][key] = self._LegacyCubeLoader_(val)

        self.metas.append(meta)
