"""Module for translating PyTorch convolutional layers to PyDTNN equivalent layers."""

import logging
from typing import Any

# Typing related (or non important) imports
import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.layers.conv_2d import Conv2D

__all__ = ("Conv2d",)

logger = logging.getLogger(__name__)


# Functionality imports


def Conv2d(args: dict[str, Any]) -> Conv2D:
    """
    Converts a PyTorch Conv2d layer configuration to a PyDTNN Conv2D layer instance.

    Args:
        args: A dictionary containing the PyTorch layer configuration and arguments.

    Returns:
        An initialized PyDTNN Conv2D layer.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

    # PyTorch attributes:
    # Not used: in channels, device, dtype
    pytorch_kernel_size = "kernel_size"  # INT or Tuple[INT, INT]
    pytorch_stride = "stride"  # INT or Tuple[INT, INT]
    pytorch_padding = "padding"  # INT or Tuple[INT, INT]
    pytorch_dilation = "dilation"  # INT
    pytorch_groups = "groups"  # INT
    pytorch_bias = "bias"  # BOOL
    pytorch_ouput_channels = "out_channels"
    # PYTORCH_PADDING_MODE = "padding_mode" # STRING. Values: {"zeros",
    # "reflect", "replicate", "circular"} | In PyDTNN "zeros" is the only
    # implemented
    torch_dict_keys = [
        pytorch_kernel_size,
        pytorch_stride,
        pytorch_padding,
        pytorch_dilation,
        pytorch_groups,
        pytorch_bias,
        pytorch_ouput_channels,
    ]

    # PyDTNN attributes:
    pydtnn_filter_shape = "filter_shape"
    pydtnn_stride = "stride"
    pydtnn_padding = "padding"
    pydtnn_dilation = "dilation"
    pydtnn_nfilters = "nfilters"
    pydtnn_use_bias = "use_bias"
    pydtnn_dict_keys = [
        pydtnn_filter_shape,
        pydtnn_stride,
        pydtnn_padding,
        pydtnn_dilation,
        pydtnn_nfilters,
        pydtnn_use_bias,
        pydtnn_nfilters,
    ]
    # Not used: "activation" and "grouping"
    # Used, but in other place: "weights_initializer", "biases_initializer"

    layer_args = cm.prepare_pydtnn_arguments(
        arguments=args[cm.ARGUMENTS],
        torch_dict_keys=torch_dict_keys,
        pydtnn_dict_keys=pydtnn_dict_keys,
    )

    if pydtnn_filter_shape in layer_args:
        pool_shape = layer_args[pydtnn_filter_shape]
        if isinstance(pool_shape, int):
            layer_args[pydtnn_filter_shape] = (pool_shape, pool_shape)
        # else: It must be a Tuple[int, int], so it's okay
    # else: Nothing special

    return Conv2D(**layer_args)
