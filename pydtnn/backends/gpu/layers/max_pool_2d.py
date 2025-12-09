from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.backends.gpu.layers.abstract.pool_2d_layer import AbstractPool2DLayerGPU
from pydtnn.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


class MaxPool2DGPU(MaxPool2D[TensorGPU], AbstractPool2DLayerGPU):

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        self.initialize_pool_2d_gpu(prev_shape, x, pool_mode)
