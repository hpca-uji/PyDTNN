from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.backends.pycuda.layers.abstract.pool_2d_layer import AbstractPool2DLayerPycuda
from pydtnn.libs import cudnn as cudnn
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


class MaxPool2DPycuda(MaxPool2D[TensorGPU], AbstractPool2DLayerPycuda):

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        self.initialize_pool_2d_gpu(prev_shape, x, pool_mode)
