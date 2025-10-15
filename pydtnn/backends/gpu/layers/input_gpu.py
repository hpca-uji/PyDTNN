# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray

from pydtnn.layers import Input
from pydtnn.backends.gpu.layers.layer_gpu import LayerGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.types import shape_t

class InputGPU(LayerGPU, Input):

    def initialize(self, prev_shape: shape_t, x: TensorGPU):
        super().initialize(prev_shape, x)
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

    def forward(self, x: TensorGPU) -> TensorGPU:
        return x

    def backward(self, dy: TensorGPU) -> TensorGPU:
        return dy
