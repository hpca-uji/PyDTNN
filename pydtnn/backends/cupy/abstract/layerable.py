"""CuPy-based implementation of layerable components for the PyDTNN framework."""

from pydtnn.backends.cupy.abstract.base import BaseCupy
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy


__all__ = ("LayerableCupy",)


class LayerableCupy(LayerableNumpy, BaseCupy):
    """Abstract base class for layers utilizing CuPy as the backend."""
