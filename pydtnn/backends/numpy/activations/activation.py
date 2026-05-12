"""
NumPy backend implementation for activation layers in PyDTNN.
"""
import logging
from typing import TYPE_CHECKING

from pydtnn.activations.activation import Activation
from pydtnn.backends.numpy.abstract.layerable import LayerableNumpy
from pydtnn.libs import numpy as np

__all__ = ("ActivationNumpy",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import numpy as np


class ActivationNumpy(Activation[np.ndarray], LayerableNumpy):
    """
    Base class for activation layers using the NumPy backend.
    """
    ...