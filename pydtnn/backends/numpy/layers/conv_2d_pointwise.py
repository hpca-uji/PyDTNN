"""NumPy backend implementation for pointwise 2D convolution layers."""

import logging
from typing import TYPE_CHECKING

from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.libs import numpy as np
__all__ = ("Conv2DPointwiseNumpy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class Conv2DPointwiseNumpy(Conv2DPointwise[np.ndarray], Conv2DNumpy):
    """NumPy-based implementation of a pointwise 2D convolution layer."""
