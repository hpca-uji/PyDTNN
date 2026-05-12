"""
Abstract base class for Winograd convolution backends in PyDTNN.
"""

from pydtnn.backends.numpy.abstract.base import BaseNumpy

__all__ = ("BaseWinograd",)


class BaseWinograd(BaseNumpy):
    """
    Base class providing the interface for Winograd-based convolution operations.
    """

    ...
