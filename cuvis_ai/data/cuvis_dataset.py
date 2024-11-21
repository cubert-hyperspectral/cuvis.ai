import cuvis
from pathlib import Path
from typing import Optional, Callable
from torchvision import tv_tensors
from .OutputFormat import OutputFormat
import torch
from torchvision.datasets import VisionDataset
import yaml
from .MetadataUtils import metadataInit
from ..tv_transforms import WavelengthList
from pycocotools.coco import COCO
import copy
from .Labels2TV import convert_COCO2TV

from cuvis.General import SDKException

EXTENSION_SESSION = '.cu3s'
EXTENSION_LEGACY = '.cu3'

CUVIS_NON_CUBE_REFERENCES = (
    cuvis.ReferenceType.Distance, cuvis.ReferenceType.SpRad)


class _SessionCubeLoader:
    def __init__(self, path: Path, idx: int, proc_mode: Optional[cuvis.ProcessingMode] = None):
        self.path = path
        self.idx = idx
        self.proc_mode = proc_mode

    def __call__(self):
        sess = cuvis.SessionFile(str(self.path))
        mesu = sess[self.idx]
        need_reprocess = bool(self.proc_mode is None)

        if mesu.cube is None:
            need_reprocess = True

        if need_reprocess and self.proc_mode is not None:
            pc = cuvis.ProcessingContext(sess)
            pc.processing_mode = self.proc_mode
            mesu = pc.apply(mesu)

        if mesu.cube is None:
            raise ValueError(f"Could not load Cube idx={self.idx} from SessionFile {self.path}.")  # nopep8
        cube = tv_tensors.Image(mesu.cube)
        return cube.to(memory_format=torch.channels_last)


class _SessionReferenceLoader:
    def __init__(self, path: Path, reftype: cuvis.ReferenceType):
        self.path = path
        self.reftype = reftype

    def __call__(self):
        sess = cuvis.SessionFile(str(self.path))
        mesu = sess.get_reference(0, self.reftype)

        if mesu.cube is None:
            raise ValueError(f"Could not load Reference Cube {self.reftype} from SessionFile {self.path}.")  # nopep8
        cube = tv_tensors.Image(mesu.cube)
        return cube.to(memory_format=torch.channels_last)


class _LegacyCubeLoader:
    def __init__(self, path: Path, proc_mode: Optional[cuvis.ProcessingMode] = None):
        self.path = path
        self.proc_mode = proc_mode

    def __call__(self):
        mesu = cuvis.Measurement(str(self.path))
        need_reprocess = bool(self.proc_mode is None)

        if mesu.cube is None:
            need_reprocess = True

        if need_reprocess and self.proc_mode is not None:
            pc = cuvis.ProcessingContext(mesu)
            pc.processing_mode = self.proc_mode
            mesu = pc.apply(mesu)

        if mesu.cube is None:
            raise ValueError(f"Could not load Cube from Legacy Measurement {self.path}.")  # nopep8
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
        tmp_mesu = session[0]
        tmp_cube = tmp_mesu.cube

        sess_meta = metadataInit(session_path, self.fileset_metadata)

        sess_meta["shape"] = (
            tmp_cube.width, tmp_cube.height, tmp_cube.channels)
        canvas_size = (sess_meta["shape"][0], sess_meta["shape"][1])

        sess_meta["wavelengths_nm"] = WavelengthList(
            tmp_cube.wavelength)

        try:
            sess_meta["framerate"] = session.fps
        except SDKException:
            pass

        sess_meta.setdefault('references', {})
        for ref in cuvis.ReferenceType:
            if ref.name in sess_meta['references']:
                continue

            try:
                refmesu = session.get_reference(0, ref)
            except SDKException:
                refmesu = None

            if refmesu is None or ref in CUVIS_NON_CUBE_REFERENCES:
                continue

            sess_meta['references'][ref.name] = _SessionReferenceLoader(
                session_path, ref)

        label_path = session_path.with_suffix('.json')
        coco = None
        if label_path.exists():
            coco = COCO(str(label_path))
            ids = list(sorted(coco.imgs.keys()))

        for idx in range(len(session)):
            cube_path = F"{session_path}:{idx}"
            self.paths.append(cube_path)
            self.cubes.append(_SessionCubeLoader(
                session_path, idx, self.processing_mode))

            meta = copy.deepcopy(sess_meta)

            for k, v in meta["references"].items():
                if not isinstance(v, str):
                    continue
                if Path(v).suffix == EXTENSION_SESSION:
                    meta['references'][k] = _SessionCubeLoader(
                        v, 0, cuvis.ProcessingMode.Raw)
                elif Path(v).suffix == EXTENSION_LEGACY:
                    meta['references'][k] = _LegacyCubeLoader(
                        v, cuvis.ProcessingMode.Raw)

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
        self.paths.append(str(legacy_path))
        label_path = legacy_path.with_suffix(".json")

        if self.metadata_filepath:
            meta = Metadata(legacy_path, self.fileset_metadata)
        else:
            meta = Metadata(legacy_path)

        mesu = cuvis.Measurement(str(legacy_path))
        cube = mesu.cube

        meta["shape"] = (cube.width,
                         cube.height, cube)
        meta["wavelengths_nm"] = WavelengthList(cube.wavelength)

        l = None
        canvas_size = (meta["shape"][0], meta["shape"][1])
        if label_path.exists():
            coco = COCO(label_path)
            anns = coco.loadAnns(coco.getAnnIds(list(coco.imgs.keys())[0]))[0]
            try:
                anns["wavelength"] = coco.imgs[0]["wavelength"]
            except KeyError:
                pass

            l = convert_COCO2TV(anns, canvas_size)
        self.labels.append(l)

        self.cubes.append(_LegacyCubeLoader(
            legacy_path, self.processing_mode))

        meta["integration_time_us"] = int(mesu.integration_time * 1000)
        meta["flags"] = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "Flag_" in key]:
            meta["flags"][key] = val
        meta["references"] = {}
        for key, val in [(key, mesu.data[key]) for key in mesu.data.keys() if "_ref" in key]:
            meta["references"][key] = _LegacyCubeLoader(
                val, cuvis.ProcessingMode.Raw)

        self.metas.append(meta)

    def _clear(self):
        self.paths = []
        self.cubes = []
        self.metas = []
        self.labels = []
        self.fileset_metadata = {}
        self.initialized = False
