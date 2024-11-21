import os
import cuvis
import numpy as np
import copy
from typing import Optional, Callable
import torch
from torchvision import tv_tensors
from pycocotools.coco import COCO

from cuvis.General import SDKException

from .Labels2TV import convert_COCO2TV
from .MetadataUtils import metadataInit
from .NumpyDataSet import NumpyDataSet
from .OutputFormat import OutputFormat
from ..tv_transforms import WavelengthList
from pathlib import Path

debug_enabled = True


EXTENSION_SESSION = '.cu3s'
EXTENSION_LEGACY = '.cu3'

CUVIS_NON_CUBE_REFERENCES = (
    cuvis.ReferenceType.Distance, cuvis.ReferenceType.SpRad)


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

    class _SessionCubeLoader_:
        def __init__(self, path, idx, proc_mode=None):
            self.path = path
            self.idx = idx
            self.proc_mode = proc_mode

        def __call__(self, to_dtype: np.dtype):
            sess = cuvis.SessionFile(self.path)
            mesu = sess.get_measurement(self.idx)
            need_reprocess = bool(self.proc_mode is None)
            try:
                cube = mesu.data["cube"].array
            except KeyError:
                need_reprocess = True

            if need_reprocess:
                pc = cuvis.ProcessingContext(sess)
                if self.proc_mode is not None:
                    pc.processing_mode = self.proc_mode
                mesu = pc.apply(mesu)

            cube = mesu.data["cube"].array

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
            try:
                cube = cuvis.SessionFile(self.path).get_reference(
                    0, self.reftype).data["cube"].array
            except KeyError:
                sess = cuvis.SessionFile(self.path)
                mesu = sess.get_reference(0, self.reftype)
                pc = cuvis.ProcessingContext(sess)
                pc.processing_mode = cuvis.ProcessingMode.Raw
                mesu = pc.apply(mesu)
                cube = mesu.data["cube"].array

            if cube.dtype != to_dtype:
                cube = cube.astype(to_dtype)
            cube = tv_tensors.Image(cube)
            while len(cube.shape) < 4:
                cube = cube.unsqueeze(0)
            return cube.to(memory_format=torch.channels_last)

    class _LegacyCubeLoader_:
        def __init__(self, path, proc_mode=None):
            self.path = path
            self.proc_mode = proc_mode

        def __call__(self, to_dtype: np.dtype):
            mesu = cuvis.Measurement.load(self.path)
            need_reprocess = bool(self.proc_mode is None)
            try:
                cube = mesu.data["cube"].array
            except KeyError:
                need_reprocess = True

            if need_reprocess:
                pc = cuvis.ProcessingContext(mesu)
                if self.proc_mode is not None:
                    pc.processing_mode = self.proc_mode
                mesu = pc.apply(mesu)

            cube = mesu.data["cube"].array

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
                 force_proc_mode: Optional[cuvis.ProcessingMode] = None
                 ):
        self.processing_mode = force_proc_mode
        super().__init__(root, transforms=transforms, transform=transform,
                         target_transform=target_transform, output_format=output_format, output_lambda=output_lambda)

    def _load_directory(self, dir_path: str):
        dir_path = Path(dir_path)
        if debug_enabled:
            print("Reading from directory:", dir_path)
        fileset_session = dir_path.glob(f'**/*{EXTENSION_SESSION}')

        fileset_legacy = dir_path.glob(f'**/*{EXTENSION_LEGACY}')

        for cur_path in fileset_session:
            self._load_session_file(cur_path)
        for cur_path in fileset_legacy:
            self._load_legacy_file(cur_path)

    def _load_session_file(self, filepath: Path):
        if debug_enabled:
            print("Found file:", filepath)
        labelpath = filepath.with_suffix('.json')

        crt_session = cuvis.SessionFile(str(filepath))

        cube_count = len(crt_session)
        if debug_enabled:
            print("Session file has", cube_count, "cubes")

        sess_meta = metadataInit(filepath, self.fileset_metadata)

        temp_mesu = crt_session.get_measurement(0)
        temp_cube = temp_mesu.cube

        sess_meta["shape"] = (temp_cube.width,
                              temp_cube.height, temp_cube.channels)
        canvas_size = (sess_meta["shape"][0], sess_meta["shape"][1])
        sess_meta["wavelengths_nm"] = WavelengthList(
            temp_cube.wavelength)
        try:
            sess_meta["framerate"] = crt_session.fps
        except SDKException:
            pass

        sess_meta.setdefault('references', {})

        for reftype in cuvis.ReferenceType:
            if reftype.name in sess_meta['references']:
                continue

            try:
                refmesu = crt_session.get_reference(0, reftype)
            except SDKException:
                refmesu = None

            if refmesu is None or reftype in CUVIS_NON_CUBE_REFERENCES:
                continue

            sess_meta["references"][reftype.name] = self._SessionReferenceLoader_(
                str(filepath), reftype)

        coco = None
        if labelpath.exists():
            coco = COCO(str(labelpath))
            ids = list(sorted(coco.imgs.keys()))

        for idx in range(cube_count):
            cube_path = F"{filepath}:{idx}"
            self.paths.append(cube_path)
            self.cubes.append(self._SessionCubeLoader_(
                str(filepath), idx, self.processing_mode))

            meta = copy.deepcopy(sess_meta)

            for k, v in meta["references"].items():
                if not isinstance(v, str):
                    continue
                if Path(v).suffix == EXTENSION_SESSION:
                    meta['references'][k] = self._SessionCubeLoader_(
                        str(v), 0, cuvis.ProcessingMode.Raw)
                elif Path(v).suffix == EXTENSION_LEGACY:
                    meta['references'][k] = self._LegacyCubeLoader_(
                        str(v), cuvis.ProcessingMode.Raw)

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

    def _load_legacy_file(self, filepath: Path):
        if debug_enabled:
            print("Found file:", filepath)
        self.paths.append(filepath)
        labelpath = filepath.with_suffix(".json")

        if self.metadata_filepath:
            meta = Metadata(filepath, self.fileset_metadata)
        else:
            meta = Metadata(filepath)

        mesu = cuvis.Measurement(filepath)
        cube = mesu.cube

        meta["shape"] = (cube.width,
                         cube.height, cube)
        meta["wavelengths_nm"] = WavelengthList(cube.wavelength)

        l = None
        canvas_size = (meta["shape"][0], meta["shape"][1])
        if labelpath.exists():
            coco = COCO(labelpath)
            anns = coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0]))[0]
            try:
                anns["wavelength"] = coco.imgs[0]["wavelength"]
            except KeyError:
                pass

            l = convert_COCO2TV(anns, canvas_size)
        self.labels.append(l)

        self.cubes.append(self._LegacyCubeLoader_(
            filepath, self.processing_mode))

        meta["integration_time_us"] = int(mesu.integration_time * 1000)
        meta["flags"] = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta["flags"][key] = val
        meta["references"] = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            meta["references"][key] = self._LegacyCubeLoader_(
                val, cuvis.ProcessingMode.Raw)

        self.metas.append(meta)
