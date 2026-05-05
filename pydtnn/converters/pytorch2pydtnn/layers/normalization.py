# In this file must be implemented only the translation of PyTorch Normalization layers to its PyDTNN equivalent.

import logging
from typing import Any

import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.layers.batch_normalization import BatchNormalization

__all__ = (
    "BatchNorm2d",
)

logger = logging.getLogger(__name__)

# Typing related (or non important) imports

# Functionality imports


def BatchNorm2d(args: dict[str, Any]) -> BatchNormalization:
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d

    # PyTorch attributes:
    # Not used: num_features, affine, track_running_stats
    PYTORCH_EPS = "eps"  # Float
    PYTORCH_MOMENTUM = "momentum"  # Float

    torch_dict_keys = [PYTORCH_MOMENTUM, PYTORCH_EPS]

    # PyDTNN attributes:
    # Not used: beta, gamma
    PYDTNN_MOMENTUM = "momentum"
    PYDTNN_EPSILON = "epsilon"

    pydtnn_dict_keys = [PYDTNN_MOMENTUM, PYDTNN_EPSILON]

    layer_args = cm.prepare_pydtnn_arguments(arguments=args[cm.ARGUMENTS], torch_dict_keys=torch_dict_keys, pydtnn_dict_keys=pydtnn_dict_keys)

    return BatchNormalization(**layer_args)
