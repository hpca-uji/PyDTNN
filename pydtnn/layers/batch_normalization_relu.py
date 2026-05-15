from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.utils.constants import Array
from pydtnn.backends.fuse.layers.layer import LayerFuse as FusedLayerMixIn

__all__ = (
    "BatchNormalizationRelu",
)

class BatchNormalizationRelu[T: Array](FusedLayerMixIn[T], BatchNormalization[T]):
    """
    Abstract base class for fused Batch Normalization and ReLU layers.
    """

    pass
