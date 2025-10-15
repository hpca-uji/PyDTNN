# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.layers import AveragePool2D
from pydtnn.backends.gpu.layers.abstract_pool_2d_layer_gpu import AbstractPool2DLayerGPU
from pydtnn.backends.gpu.libs import libcudnn as cudnn
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.types import shape_t

class AveragePool2DGPU(AbstractPool2DLayerGPU, AveragePool2D):

    def initialize(self, prev_shape: shape_t, x: TensorGPU) -> TensorGPU:
        super().initialize(prev_shape, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING']
        self.initialize_pool_2d_gpu(prev_shape, x, pool_mode)
