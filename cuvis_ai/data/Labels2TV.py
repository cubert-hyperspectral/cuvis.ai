import numpy as np
from torchvision.tv_tensors import BoundingBoxes, Mask
from skimage.draw import polygon2mask
from ..tv_transforms import WavelengthList


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


def convert_COCO2TV(coco, size):
    """Helper function to convert bounding boxes and segmentation polygons to torchvision tensors."""
    out = {}

    for k, v in coco.items():
        if k == "bbox":
            # print(F"Canvas_size: {size} bbox COCO: {v}")
            out["bbox"] = BoundingBoxes(v, format="XYWH", canvas_size=size)
        elif k == "segmentation":
            try:
                out["segmentation"] = Mask(RLE2mask(v["counts"], v["size"]))
            except KeyError:
                out["segmentation"] = Mask(polygon2mask(
                    size, np.array(v[0]).reshape(-1, 2)).astype(np.uint8))
        elif k == "wavelength":
            out["wavelength"] = WavelengthList(v)
        else:
            out[k] = v
    return out
