# In this file must be implemented only the translation of PyTorch Linear
# layers to its PyDTNN equivalent.

"""Module for converting PyTorch Linear layers to PyDTNN FC layers."""

import logging
from typing import Any

# Typing related (or non important) imports
import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.layers.fc import FC

__all__ = ("Linear",)

logger = logging.getLogger(__name__)


# Functionality imports


def Linear(args: dict[str, Any]) -> FC:
    """
    Converts a PyTorch Linear layer configuration to a PyDTNN FC layer.

    Args:
        args: A dictionary containing the PyTorch layer configuration.

    Returns:
        An initialized PyDTNN FC layer instance.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear

    # PyTorch attributes:
    # Not used: in_features (It's not used due the way the layer's initialization works in PyDTNN)
    pytorch_bias = "bias"
    pytorch_out_features = "out_features"
    torch_dict_keys = [pytorch_bias, pytorch_out_features]

    # PyDTNN attributes:
    # Not used: activation
    # Used, but in a different place: weights_initializer, biases_initializer
    pydtnn_bias = "use_bias"
    pydtnn_shape = "shape"
    pydtnn_dict_keys = [pydtnn_bias, pydtnn_shape]

    layer_args = cm.prepare_pydtnn_arguments(
        arguments=args[cm.ARGUMENTS],
        torch_dict_keys=torch_dict_keys,
        pydtnn_dict_keys=pydtnn_dict_keys,
    )

    initializers = cm.set_initializer_with_pytorch_values(args[cm.ARGUMENTS]["_parameters"], transpose_values = True)
    layer_args.update(initializers)

    # PyDTNN expects the shape as a tuple instead of an int.
    if pydtnn_shape in layer_args and isinstance(layer_args[pydtnn_shape], int):
        layer_args[pydtnn_shape] = (layer_args[pydtnn_shape],)

    return FC(**layer_args)
