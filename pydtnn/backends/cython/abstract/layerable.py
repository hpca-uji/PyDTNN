"""
Cython-based abstract layerable module for PyDTNN.
"""

from pydtnn.backends.cython.abstract.base import BaseCython
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy

__all__ = ("LayerableCython",)


class LayerableCython(LayerableNumpy, BaseCython):
    """
    Abstract base class for Cython-accelerated layers, inheriting from both
    Numpy-based layerable interfaces and Cython base functionality.
    """

    ...
