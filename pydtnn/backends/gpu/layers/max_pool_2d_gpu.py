import pycuda.gpuarray as gpuarray  #type: ignore

from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.backends.gpu.layers.abstract_pool_2d_layer_gpu import AbstractPool2DLayerGPU
from pydtnn.backends.gpu.libs import libcudnn as cudnn
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.types import ArrayShape


class MaxPool2DGPU(AbstractPool2DLayerGPU, MaxPool2D[TensorGPU]):

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)
        pool_mode = cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX']
        self.initialize_pool_2d_gpu(prev_shape, x, pool_mode)
