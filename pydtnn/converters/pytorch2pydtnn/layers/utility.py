"""This module provides utilities for converting PyTorch convolutional layers to their PyDTNN equivalents."""

from pydtnn.layers.flatten import Flatten as Flatten_PyDTNN
from pydtnn.layers.input import Input

__all__ = ("Flatten",)


def Identity(_args: dict[str, str]) -> Input:
    """
    Converts a PyTorch Identity layer to a PyDTNN Input layer (the one with the most similar behaviour).

    Args:
        args: A dictionary containing the configuration arguments from the PyTorch layer (they will be ignored).

    Returns:
        An initialized PyDTNN Input layer instance.
    """
    # https://docs.pytorch.org/docs/2.13/generated/torch.nn.Identity.html

    return Input()


def Flatten(args: dict[str, str]) -> Flatten_PyDTNN:
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

    return Flatten_PyDTNN()
