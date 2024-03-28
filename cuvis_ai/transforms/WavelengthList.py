from torchvision.tv_tensors import TVTensor
import torch
import numpy as np
from typing import Union, Optional, Any

class WavelengthList(TVTensor):
    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ):
        if not isinstance(data, list):
            raise ValueError("WavelengthList got invalid input data!")
        #data = np.array([float(wl) for wl in data]).reshape((len(data), 1, 1))
        #print("WLL init data:", data)
        data = np.array(data)#.reshape((len(data), 1, 1))
        tensor = cls._to_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        #print("WLL init tensor:", data)
        return tensor.as_subclass(cls)

    def __repr__(self, *, tensor_contents: Any = None) -> str:  # type: ignore[override]
        return self._make_repr()
