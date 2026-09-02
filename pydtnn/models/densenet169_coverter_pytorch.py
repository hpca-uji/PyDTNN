"""Get a Resnet50 from PyTorch converted to PyDTNN."""

from collections.abc import Sequence

import torch
import torchvision.models as torch_models

from pydtnn.abstract.layerable import Layerable
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.pytorch import from_pytorch

__all__ = ("densenet169_coverter_pytorch",)


def densenet169_coverter_pytorch(
    input_shape: ArrayShape, output_shape: ArrayShape
) -> Sequence[Layerable]:
    """Returns a PyDTNN conversion from a PyTorch's Resnet50."""

    torch_model = torch_models.densenet169(weights=torch_models.DenseNet169_Weights.IMAGENET1K_V1)
    torch_model.classifier = torch.nn.Sequential(  # pyright: ignore[reportAttributeAccessIssue]
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(
            in_features=torch_model.classifier.in_features, out_features=output_shape[0]
        ),
        torch.nn.LogSoftmax(),
    )

    return from_pytorch(input_shape, torch_model)
