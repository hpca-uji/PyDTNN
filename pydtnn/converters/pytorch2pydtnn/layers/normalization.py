# In this file must be implemented only the translation of PyTorch
# Normalization layers to its PyDTNN equivalent.

"""Module for converting PyTorch normalization layers to PyDTNN equivalents."""

import logging
from typing import Any

import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.layers.batch_normalization import BatchNormalization

__all__ = ("BatchNorm2d",)

logger = logging.getLogger(__name__)

# Typing related (or non important) imports

# Functionality imports


def BatchNorm2d(args: dict[str, Any]) -> BatchNormalization:
    """
    Converts a PyTorch BatchNorm2d layer configuration to a PyDTNN BatchNormalization layer.

    Args:
        args: A dictionary containing the PyTorch layer arguments.

    Returns:
        An initialized PyDTNN BatchNormalization layer.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d

    # PyTorch attributes:
    # Not used: num_features, affine, track_running_stats
    pytorch_eps = "eps"  # Float
    pytorch_momentum = "momentum"  # Float

    torch_dict_keys = [pytorch_momentum, pytorch_eps]

    # PyDTNN attributes:
    # Not used: beta, gamma
    pydtnn_momentum = "momentum"
    pydtnn_epsilon = "epsilon"

    pydtnn_dict_keys = [pydtnn_momentum, pydtnn_epsilon]

    layer_args = cm.prepare_pydtnn_arguments(
        arguments=args[cm.ARGUMENTS],
        torch_dict_keys=torch_dict_keys,
        pydtnn_dict_keys=pydtnn_dict_keys,
    )

    return BatchNormalization(**layer_args)
