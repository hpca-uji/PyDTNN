# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.layers import AveragePool2D
from .abstract_pool_2d_layer_gpu import AbstractPool2DLayerGPU
from ..libs import libcudnn as cudnn
from ..tensor_gpu import TensorGPU

class AveragePool2DGPU(AbstractPool2DLayerGPU, AveragePool2D):

    def initialize(self, prev_shape:tuple[int, ...], x: TensorGPU) -> TensorGPU:
        super().initialize(prev_shape, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING']
        self.initialize_pool_2d_gpu(prev_shape, x, pool_mode)
