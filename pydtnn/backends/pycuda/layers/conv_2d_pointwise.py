"""PyCUDA implementation of a 2D pointwise convolution layer."""

import logging

from pydtnn.backends.pycuda.layers.conv_2d import Conv2DPycuda
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.utils.tensor_array import TensorArray

__all__ = ("Conv2DPointwisePycuda",)

logger = logging.getLogger(__name__)


class Conv2DPointwisePycuda(Conv2DPointwise[TensorArray], Conv2DPycuda):
    """PyCUDA-accelerated 2D pointwise convolution layer (1x1 kernel)."""
