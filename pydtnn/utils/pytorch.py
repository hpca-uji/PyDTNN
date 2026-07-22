"""Get a model from PyTorch converted to PyDTNN."""

from collections.abc import Sequence

import torch

from pydtnn.abstract.layerable import Layerable
from pydtnn.converters.pytorch2pydtnn.model_convertor import get_layers_from_torch
from pydtnn.utils.constants import ArrayShape


def from_pytorch(
    input_shape: ArrayShape,
    torch_model: torch.nn.Module,
) -> Sequence[Layerable]:
    """Returns a PyDTNN conversion from a PyTorch model.

    Args:
        input_shape (ArrayShape): Data's input shape.
        output_shape (ArrayShape): The output's shape.
        torch_model_func (Callable[[ArrayShape], torch.nn.Module]): A function that will provide a PyTorch model.
    """
    layers = get_layers_from_torch(model=torch_model, input_shape=input_shape)

    return layers
