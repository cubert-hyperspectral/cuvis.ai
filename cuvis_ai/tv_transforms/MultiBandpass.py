from torchvision.transforms.v2 import Transform
import torch
from typing import Dict, Any, List
from .WavelengthList import WavelengthList
from .Bandpass import Bandpass


class MultiBandpass(Transform):
    """Apply multiple bandpasses in parallel to the input data.
    Selectively extract non-consecutive channels from the input data.
    This preprocessor node describes operations such as:
    Extract channels 4 to 10 and 30 to 39 and concatenate them.

    Parameters
    ----------
    bandpasses : List(Bandpass)
        A list of :cls:`Bandpass` transformations, the output of which will be concatenated.
    """
    def __init__(self, bandpasses:List[Bandpass]):
        super().__init__()
        self.bandpasses = bandpasses
        
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if (isinstance(inpt, torch.Tensor) and len(inpt.shape) >= 4):
            # Assuming [...]NCHW dimension ordering
            channel_dim = len(inpt.shape) - 3
            bands = [bp(inpt) for bp in self.bandpasses]
            return torch.cat(bands, dim=channel_dim).as_subclass(type(inpt))
        return inpt