"""Fused layer implementation for Conv2D, BatchNormalization, and ReLU operations."""

from pydtnn.backends.fuse.layers.abstract.layer import LayerFuse as FusedLayerMixIn
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.constants import Array

__all__ = ("Conv2DBatchNormalizationRelu",)


class Conv2DBatchNormalizationRelu[T: Array](FusedLayerMixIn[T], Conv2D[T], BatchNormalization[T]):
    """Base class for fused Conv2D, BatchNormalization, and ReLU layers."""

    pass
