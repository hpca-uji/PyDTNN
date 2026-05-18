"""
Fused Batch Normalization and ReLU layer implementation.
"""

from pydtnn.backends.fuse.layers.abstract.layer import LayerFuse as FusedLayerMixIn
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.utils.constants import Array

__all__ = ("BatchNormalizationRelu",)


class BatchNormalizationRelu[T: Array](FusedLayerMixIn[T], BatchNormalization[T]):
    """
    A fused layer that combines Batch Normalization and ReLU activation.
    """

    pass
