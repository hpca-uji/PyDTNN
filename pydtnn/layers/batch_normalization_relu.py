from pydtnn.layer import FusedLayerMixIn
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.utils.constants import Array

class BatchNormalizationRelu[T: Array](FusedLayerMixIn[T], BatchNormalization[T]):
    pass