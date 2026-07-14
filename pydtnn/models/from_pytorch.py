


from collections.abc import Sequence

import torch

from pydtnn.abstract.layerable import Layerable
from pydtnn.converters.pytorch2pydtnn.model_convertor import get_layers_from_torch
from pydtnn.utils.constants import ArrayShape

import torchvision.models as torch_models

def _resnet(output_shape: ArrayShape) -> torch.nn.Module:
    torch_model = torch_models.resnet50(weights=torch_models.ResNet50_Weights.IMAGENET1K_V1)
    torch_model.fc = torch.nn.Linear(in_features=torch_model.fc.in_features, out_features=output_shape[0])
    return torch_model

def from_pytorch(input_shape: ArrayShape, output_shape: ArrayShape,
                 torch_model: torch.nn.Module | None = None, torch_model_method: callable = _resnet
                 ) -> Sequence[Layerable]:
    
    if torch_model is None:
        _torch_model = torch_model_method(output_shape)
    else:
        _torch_model = torch_model

    layers = get_layers_from_torch(model=_torch_model, input_shape=input_shape)

    return layers
