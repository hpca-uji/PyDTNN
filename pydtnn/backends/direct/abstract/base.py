"""Module providing the base class for direct computation backends in PyDTNN."""

from pydtnn.backends.numpy.abstract.base import BaseNumpy

__all__ = ("BaseDirect",)


class BaseDirect(BaseNumpy):
    """Abstract base class for direct computation backends, inheriting from BaseNumpy."""
