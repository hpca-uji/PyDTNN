from pydtnn.layer_base import FusedLayerMixIn
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.utils.constants import Array

class Conv2DBatchNormalization[T: Array](FusedLayerMixIn[T], Conv2D[T], BatchNormalization[T]):
    pass
