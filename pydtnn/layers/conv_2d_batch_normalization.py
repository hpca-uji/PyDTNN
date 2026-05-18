"""
Module for fused 2D Convolution and Batch Normalization layers.
"""

from pydtnn.backends.fuse.layers.layer import LayerFuse as FusedLayerMixIn
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.constants import Array

__all__ = ("Conv2DBatchNormalization",)


class Conv2DBatchNormalization[T: Array](FusedLayerMixIn[T], Conv2D[T], BatchNormalization[T]):
    """
    Base class for fused 2D Convolution and Batch Normalization layers.
    """

    pass
