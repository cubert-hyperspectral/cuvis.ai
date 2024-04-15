from torchvision.transforms.v2 import Transform
import torch
from typing import Dict, Any
from .WavelengthList import WavelengthList


class Bandpass(Transform):
    """Apply a bandpass operation over the input.
    Selectively extract channels from the input data.
    Assumes the input data is in NCHW memory format.
    
    Args:
        from_channel: First channel to extract.
        to_channel (optional): Last channel to extract. If ommited, only the first channel is extracted.
    """
    
    def __init__(self, from_channel:int, to_channel:int=None):
        super().__init__()
        self.from_channel = from_channel
        if to_channel is None: # Using only one channel
            self.to_channel = from_channel
        else:
            self.to_channel = to_channel
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if (isinstance(inpt, torch.Tensor) and len(inpt.shape) >= 4):
            # Assuming [...]NCHW dimension ordering
            channel_dim = len(inpt.shape) - 3
            channels = torch.split(inpt, 1, dim=channel_dim)
            return torch.cat(channels[self.from_channel:self.to_channel + 1], dim=channel_dim).as_subclass(type(inpt))
        return inpt