import cuvis
from pathlib import Path
from typing import Optional, Callable
from torchvision import tv_tensors
from .OutputFormat import OutputFormat
import torch
from torchvision.datasets import VisionDataset
import yaml
from ..tv_transforms import WavelengthList
from pycocotools.coco import COCO
import copy
from .Labels2TV import convert_COCO2TV
import numpy as np
from .metadata import Metadata, get_meta_from_session, get_meta_from_mesu, get_meta_from_path
from functools import lru_cache, partial

from cuvis.General import SDKException

EXTENSION_SESSION = '.cu3s'
EXTENSION_LEGACY = '.cu3'

CUVIS_NON_CUBE_REFERENCES = (
    cuvis.ReferenceType.Distance, cuvis.ReferenceType.SpRad)


@lru_cache
def get_session_cube(path, idx, proc_mode):
    sess = cuvis.SessionFile(str(path))
    mesu = sess[idx]
    need_reprocess = bool(proc_mode is None)

    if mesu.cube is None:
        need_reprocess = True

    if need_reprocess and proc_mode is not None:
        pc = cuvis.ProcessingContext(sess)
        pc.processing_mode = proc_mode
        mesu = pc.apply(mesu)

    if mesu.cube is None:
        raise ValueError(f"Could not load Cube idx={idx} from SessionFile {path}.")  # nopep8
    cube = tv_tensors.Image(mesu.cube)
    return cube.to(memory_format=torch.channels_last)


@lru_cache
def get_session_reference(path, reftype):
    sess = cuvis.SessionFile(str(path))
    mesu = sess.get_reference(0, reftype)

    if mesu.cube is None:
        raise ValueError(f"Could not load Reference Cube {reftype} from SessionFile {path}.")  # nopep8
    cube = tv_tensors.Image(mesu.cube)
    return cube.to(memory_format=torch.channels_last)


@lru_cache
def get_legacy_cube(path, proc_mode):
    mesu = cuvis.Measurement(str(path))
    need_reprocess = bool(proc_mode is None)

    if mesu.cube is None:
        need_reprocess = True

    if need_reprocess and proc_mode is not None:
        pc = cuvis.ProcessingContext(mesu)
        pc.processing_mode = proc_mode
        mesu = pc.apply(mesu)

    if mesu.cube is None:
        raise ValueError(f"Could not load Cube from Legacy Measurement {path}.")  # nopep8
    cube = tv_tensors.Image(mesu.cube)
    return cube.to(memory_format=torch.channels_last)


class CuvisDataset(VisionDataset):

    def __init__(self, root: str = None,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 output_format: OutputFormat = OutputFormat.Full,
                 output_lambda: Optional[Callable] = None,
                 force_proc_mode: Optional[cuvis.ProcessingMode] = None
                 ):
        self.processing_mode = force_proc_mode
        super().__init__(root, transforms=transforms,
                         transform=transform, target_transform=target_transform)
        self.output_format = output_format
        self.output_lambda = output_lambda

        self._clear()
        if root is None or not Path(root).exists():
            raise RuntimeError(
                "Could not find root directory.")

        self.root_dir = Path(root)

        self.metadata_path = self.root_dir / "metadata.yaml"
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.fileset_metadata = yaml.safe_load(f)
        else:
            self.metadata_path = None

        self._load_directory(self.root_dir)
        self.initialized = True

    def _load_directory(self, directory: Path):

        fileset_session = directory.glob(f'**/*{EXTENSION_SESSION}')
        fileset_legacy = directory.glob(f'**/*{EXTENSION_LEGACY}')

        for path in fileset_session:
            self._load_session_file(path)
        for path in fileset_legacy:
            self._load_legacy_file(path)

    def _load_session_file(self, session_path: Path):
        session = cuvis.SessionFile(str(session_path))

        if self.processing_mode is None:
            tmp_mesu = session[0]
            self.processing_mode = tmp_mesu.processing_mode

        sess_meta = get_meta_from_session(session, session_path)

        canvas_size = (sess_meta.shape[0], sess_meta.shape[1])

        label_path = session_path.with_suffix('.json')
        coco = None
        if label_path.exists():
            coco = COCO(str(label_path))
            ids = list(sorted(coco.imgs.keys()))

        for idx in range(len(session)):
            cube_path = F"{session_path}:{idx}"
            self.paths.append(cube_path)
            self.cubes.append(partial(get_session_cube,
                                      str(session_path), idx, self.processing_mode))

            meta = copy.deepcopy(sess_meta)

            for k, v in meta.references.items():
                if not isinstance(v, str):
                    continue
                if Path(v).suffix == EXTENSION_SESSION and v == str(session_path):
                    meta.references[k] = partial(get_session_reference,
                                                 str(v), k)
                if Path(v).suffix == EXTENSION_SESSION:
                    meta.references[k] = partial(get_session_cube,
                                                 str(v), 0, cuvis.ProcessingMode.Raw)
                elif Path(v).suffix == EXTENSION_LEGACY:
                    meta.references[k] = partial(get_legacy_cube,
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

    def _load_legacy_file(self, legacy_path: Path):
        self.paths.append(legacy_path)
        labelpath = legacy_path.with_suffix(".json")

        mesu = cuvis.Measurement(legacy_path)

        meta = get_meta_from_mesu(mesu)

        canvas_size = (meta.shape[0], meta.shape[1])

        l = None
        if labelpath.exists():
            coco = COCO(labelpath)
            anns = coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0]))[0]
            try:
                anns["wavelength"] = coco.imgs[0]["wavelength"]
            except KeyError:
                pass

            l = convert_COCO2TV(anns, canvas_size)
        self.labels.append(l)

        self.cubes.append(partial(get_legacy_cube,
                                  legacy_path, self.processing_mode))

        meta.flags = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta.flags[key] = val
        meta.references = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            meta.references[key] = partial(get_legacy_cube,
                                           val, cuvis.ProcessingMode.Raw)

        self.metas.append(meta)

    def _clear(self):
        self.paths = []
        self.cubes = []
        self.metas = []
        self.labels = []
        self.fileset_metadata = {}
        self.initialized = False
