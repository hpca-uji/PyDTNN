"""Numpy backend implementation for layerable components in PyDTNN."""

import numpy as np

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.numpy.abstract.base import BaseNumpy

__all__ = ("LayerableNumpy",)


class LayerableNumpy(Layerable[np.ndarray], BaseNumpy):
    """Numpy-specific implementation of a layerable component supporting distributed operations."""
