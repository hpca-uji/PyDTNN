"""Cython backend abstract base module for PyDTNN."""

from pydtnn.backends.numpy.abstract.base import BaseNumpy

__all__ = ("BaseCython",)


class BaseCython(BaseNumpy):
    """Abstract base class for Cython-accelerated backends in PyDTNN."""
