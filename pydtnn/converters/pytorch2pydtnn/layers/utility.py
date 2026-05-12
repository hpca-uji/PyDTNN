"""
This module provides utilities for converting PyTorch convolutional layers to their PyDTNN equivalents.
"""

import logging

from pydtnn.layers.flatten import Flatten as _Flatten

__all__ = ("Flatten",)

logger = logging.getLogger(__name__)

# Functionality imports
# import pydtnn.converters.pytorch2pydtnn.common as cm


def Flatten(args: dict[str, str]) -> _Flatten:
    """
    Converts a PyTorch Flatten layer to a PyDTNN Flatten layer.

    Args:
        args: A dictionary containing the configuration arguments from the PyTorch layer.

    Returns:
        An initialized PyDTNN Flatten layer instance.
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html#torch.nn.Flatten
    # torch.nn.Flatten(start_dim=1, end_dim=-1)

    # PyTorch attributes:
    # Not used: start_dim, end_dim (It's not used due the way the layer's initialization works in PyDTNN)
    # PyDTNN attributes: None
    # not_used = args

    return _Flatten()
