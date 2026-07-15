"""Get a Resnet50 from PyTorch converted to PyDTNN."""

from collections.abc import Sequence

import torch
import torchvision.models as torch_models

from pydtnn.abstract.layerable import Layerable
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.pytorch import from_pytorch


def resnet50_from_pytorch(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """Returns a PyDTNN conversion from a PyTorch's Resnet50."""

    torch_model = torch_models.resnet50(weights=torch_models.ResNet50_Weights.IMAGENET1K_V1)
    torch_model.fc = torch.nn.Linear(
        in_features=torch_model.fc.in_features, out_features=output_shape[0]
    )
    return from_pytorch(input_shape, torch_model)
