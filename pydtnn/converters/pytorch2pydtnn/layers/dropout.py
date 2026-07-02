# In this file must be implemented only the translation of PyTorch Dropout
# layers to its PyDTNN equivalent.

"""
Module for converting PyTorch Dropout layers to PyDTNN Dropout layers.
"""

import logging
from typing import Any

# Typing related (or non important) imports
import pydtnn.converters.pytorch2pydtnn.common as cm
from pydtnn.layers.dropout import Dropout

__all__ = ("dropout",)

logger = logging.getLogger(__name__)


# Functionality imports


def dropout(args: dict[str, Any]) -> Dropout:
    """
    Converts a PyTorch Dropout layer configuration to a PyDTNN Dropout layer.

    Args:
        args: A dictionary containing the PyTorch layer configuration.

    Returns:
        An instance of the PyDTNN Dropout layer.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout

    # PyTorch attributes:
    # Not used: inplace: Bool
    pytorch_p = "p"
    torch_dict_keys = [pytorch_p]

    # PyDTNN attributes:
    pdytnn_rate = "rate"
    pydtnn_dict_keys = [pdytnn_rate]

    layer_args = cm.prepare_pydtnn_arguments(
        arguments=args[cm.ARGUMENTS],
        torch_dict_keys=torch_dict_keys,
        pydtnn_dict_keys=pydtnn_dict_keys,
    )

    return Dropout(**layer_args)
