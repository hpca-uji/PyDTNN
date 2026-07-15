"""Get a model from PyTorch converted to PyDTNN."""

from collections.abc import Sequence
from typing import Callable

import torch
import torchvision.models as torch_models

from pydtnn.abstract.layerable import Layerable
from pydtnn.converters.pytorch2pydtnn.model_convertor import get_layers_from_torch
from pydtnn.utils.constants import ArrayShape


def _resnet(output_shape: ArrayShape) -> torch.nn.Module:
    """Returns a PyDTNN conversion from a PyTorch model."""

    torch_model = torch_models.resnet50(weights=torch_models.ResNet50_Weights.IMAGENET1K_V1)
    torch_model.fc = torch.nn.Linear(
        in_features=torch_model.fc.in_features, out_features=output_shape[0]
    )
    return torch_model


def from_pytorch(
    input_shape: ArrayShape,
    output_shape: ArrayShape,
    torch_model: torch.nn.Module | None = None,
    torch_model_func: Callable[[ArrayShape], torch.nn.Module] = _resnet,
) -> Sequence[Layerable]:
    """Returns a PyDTNN conversion from a PyTorch model.

    Args:
        input_shape (ArrayShape): Data's input shape.
        output_shape (ArrayShape): The output's shape.
        torch_model (torch.nn.Module | None): the PyTorch model to convert or
                                              None if it's going to be loaded by torch_model_func's
        torch_model_func (Callable[[ArrayShape], torch.nn.Module]): A function that will provide a PyTorch model
                                                                   (only if torch_model is None).
    """

    if torch_model is None:
        _torch_model = torch_model_func(output_shape)
    else:
        _torch_model = torch_model

    layers = get_layers_from_torch(model=_torch_model, input_shape=input_shape)

    return layers
