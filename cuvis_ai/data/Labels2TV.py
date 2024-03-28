import numpy as np
from torchvision.tv_tensors import BoundingBoxes, Mask
from skimage.draw import polygon2mask
from ..transforms import WavelengthList

def convert_COCO2TV(coco, size):
    """Helper function to convert bounding boxes and segmentation polygons to torchvision tensors."""
    out = {}

    for k, v in coco.items():
        if k == "bbox":
            #print(F"Canvas_size: {size} bbox COCO: {v}")
            out["bbox"] = BoundingBoxes(v, format="XYWH", canvas_size=size)
        elif k == "segmentation":
            out["segmentation"] = Mask(polygon2mask(size, np.array(v[0]).reshape(-1, 2)).astype(np.uint8))
        else:
            out[k] = v
    return out