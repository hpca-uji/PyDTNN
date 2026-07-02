"""Module providing the abstract base class for Winograd-compatible layers in PyDTNN."""

from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.backends.winograd.abstract.base import BaseWinograd

__all__ = ("LayerableWinograd",)


class LayerableWinograd(LayerableNumpy, BaseWinograd):
    """
    Abstract base class for layers that support Winograd convolution operations
    within the PyDTNN framework.
    """

    ...
