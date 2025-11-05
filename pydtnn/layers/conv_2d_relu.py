from pydtnn.layer import FusedLayerMixIn
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.utils.types import Array

class Conv2DRelu[T: Array](FusedLayerMixIn[T], Conv2D[T]):
    pass
