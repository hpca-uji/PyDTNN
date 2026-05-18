"""
Fused 2D Convolution and ReLU layer implementation.
"""

from pydtnn.backends.fuse.layers.layer import LayerFuse as FusedLayerMixIn
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.constants import Array

__all__ = ("Conv2DRelu",)


class Conv2DRelu[T: Array](FusedLayerMixIn[T], Conv2D[T]):
    """
    Base class for 2D Convolution layers with fused ReLU activation.
    """

    pass
