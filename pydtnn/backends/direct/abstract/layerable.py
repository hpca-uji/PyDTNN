"""
Abstract base class for direct backend layers in PyDTNN.
"""

from pydtnn.backends.direct.abstract.base import BaseDirect
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy

__all__ = ("LayerableDirect",)


class LayerableDirect(LayerableNumpy, BaseDirect):
    """
    Interface for layers compatible with the direct execution backend.
    """

    ...
