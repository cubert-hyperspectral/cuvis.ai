import torch
import torch.nn as nn
from enum import Enum


class InputDimension(Enum):
    One = 1
    Two = 2
    Three = 3
    Unknown = -1


def guess_input_dimensionalty(model: nn.Module) -> InputDimension:
    """
    Inspects the first layer of a PyTorch model to guess if the model expects
    1D (flattened), 2D (spatial), or 3D input data.

    Returns:
        InputDimension: Enum value indicating likely input dimensions.
    """
    # Get the first layer of the model
    first_layer = next(model.children(), None)

    if len(first_layer._modules.items()) > 0:
        first_layer = list(first_layer._modules.values())[0]

    # Check if there is a first layer in the model
    if first_layer is None:
        return InputDimension.Unknown  # No layers in the model

    # Analyze the type of the first layer
    if isinstance(first_layer, nn.Conv2d):
        # 2D convolution, likely expects spatial (image-like) data
        if first_layer.in_channels == 1:
            return InputDimension.Two
        else:
            return InputDimension.Three
    elif isinstance(first_layer, nn.Conv1d):
        # 1D convolution, likely expects sequence data with channels
        return InputDimension.One
    elif isinstance(first_layer, nn.Conv3d):
        # 3D convolution, likely expects 3D spatial data (e.g., video)
        return InputDimension.Three
    elif isinstance(first_layer, nn.Linear):
        return InputDimension.One  # Linear layer, likely expects flattened data

    return InputDimension.Unknown


def get_output_shape(input_shape, model):
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        output = model(dummy_input)
    return output.shape
