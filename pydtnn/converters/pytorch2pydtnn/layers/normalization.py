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
    pydtnn_momentum = "momentum"
    pydtnn_epsilon = "epsilon"
    # pydtnn_use_bias = "use_bias"

    pydtnn_dict_keys = [pydtnn_momentum, pydtnn_epsilon]

    layer_args = cm.prepare_pydtnn_arguments(
        arguments=args[cm.ARGUMENTS],
        torch_dict_keys=torch_dict_keys,
        pydtnn_dict_keys=pydtnn_dict_keys,
    )

    vars_initiaizers_transpose: dict[str, tuple[str, None | tuple[str, str]]]
    vars_initiaizers_transpose = {
        "running_mean": ("running_mean_initializer", None),
        "running_var": ("running_var_initializer", None),
    }
    initializers = cm.set_initializer_with_pytorch_values(
        args[cm.ARGUMENTS]["_buffers"], vars_initiaizers_transpose=vars_initiaizers_transpose
    )
    layer_args.update(initializers)

    vars_initiaizers_transpose = {
        "bias": ("biases_initializer", None),
        "weight": ("weights_initializer", None),
    }
    initializers = cm.set_initializer_with_pytorch_values(
        args[cm.ARGUMENTS]["_parameters"], vars_initiaizers_transpose=vars_initiaizers_transpose
    )
    layer_args.update(initializers)

    return BatchNormalization(**layer_args)
