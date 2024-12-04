import json
from dataclasses import dataclass, field
from dataclass_wizard import JSONWizard
from typing import Optional, Any
from pathlib import Path
from pycocotools.coco import COCO
from torchvision.tv_tensors import BoundingBoxes, Mask
from skimage.draw import polygon2mask
from copy import copy
import numpy as np


def RLE2mask(rle: list, mask_size: tuple) -> np.ndarray:
    mask = np.zeros(mask_size, np.uint8).reshape(-1)
    ids = 0
    value = 0
    for c in rle:
        mask[ids: ids+c] = value
        value = not value
        ids += c
    mask = mask.reshape(mask_size, order='F')
    return mask


@dataclass
class Info(JSONWizard):
    description: Optional[str] = None
    url: Optional[str] = None
    version: Optional[int] = None
    contributor: Optional[str] = None
    date_created: Optional[str] = None


@dataclass
class License(JSONWizard):
    id: int
    name: str
    url: Optional[str] = None


@dataclass
class Category(JSONWizard):
    id: int
    name: str
    supercategory: Optional[str] = None


@dataclass
class Image(JSONWizard):
    id: int
    file_name: str
    height: int
    width: int
    license: Optional[int] = None
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None
    wavelength: Optional[list[float]] = field(default_factory=list)


@dataclass
class Annotation(JSONWizard):
    id: int
    image_id: int
    category_id: int
    segmentation: Optional[list] = None
    area: Optional[float] = None
    bbox: Optional[list[float]] = None
    mask: Optional[dict[int]] = None
    iscrowd: Optional[int] = 0
    auxiliary: Optional[dict[str, Any]] = field(default_factory=dict)

    def to_torchvision(self, size):
        """Helper function to convert bounding boxes and segmentation polygons to torchvision tensors."""

        out = copy(self)
        if not self.bbox is None:
            out.bbox = BoundingBoxes(
                self.bbox, format="XYWH", canvas_size=size)
        if not self.segmentation is None:
            out.segmentation = Mask(polygon2mask(
                size, np.array(self.segmentation[0]).reshape(-1, 2)).astype(np.uint8))

        if not self.mask is None:
            out.mask = Mask(RLE2mask(self.mask["counts"], self.mask["size"]))
        return out.to_dict()


class QueryableList:
    def __init__(self, items: list[Any]):
        self._items = items

    def where(self, **conditions):
        """
        Filter items based on conditions.
        :param conditions: Keyword arguments representing field=value filters.
        :return: A new QueryableList with filtered items.
        """
        filtered_items = self._items
        for key, value in conditions.items():
            filtered_items = [
                item for item in filtered_items if getattr(item, key) == value]
        return list(filtered_items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class COCOData():

    def __init__(self, coco: COCO):
        self._coco = coco
        self._image_ids = None
        self._annotations = None
        self._images = None

    @classmethod
    def from_path(cls, path):
        return cls(COCO(str(path)))

    @property
    def image_ids(self) -> list[int]:
        if self._image_ids is None:
            self._image_ids = list(sorted(self._coco.imgs.keys()))
        return self._image_ids

    @property
    def info(self) -> Info:
        return Info.from_dict(self._coco.dataset['info'])

    @property
    def license(self) -> License:
        return Info.from_dict(self._coco.dataset['licenses'])

    @property
    def annotations(self) -> QueryableList:
        if self._annotations is None:
            self._annotations = QueryableList([Annotation.from_dict(
                v) for v in self._coco.anns.values()])
        return self._annotations

    @property
    def images(self) -> list[Image]:
        if self._images is None:
            self._images = [Image.from_dict(v)
                            for v in self._coco.imgs.values()]
        return self._images
