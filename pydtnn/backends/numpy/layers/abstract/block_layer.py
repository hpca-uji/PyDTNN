"""Module providing the abstract base class for block-based layers in the NumPy backend."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.abstract.layer import LayerNumpy
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.libs import numpy as np

__all__ = ("AbstractBlockLayerNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class AbstractBlockLayerNumpy(AbstractBlockLayer[np.ndarray], LayerNumpy):
    """Abstract base class for layers that operate on blocks using the NumPy backend."""

    pass
