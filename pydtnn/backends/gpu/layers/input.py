import pycuda.gpuarray as gpuarray  #type: ignore

from pydtnn.layers.input import Input
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


class InputGPU(Input[TensorGPU], LayerGPU):

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU):
        super().initialize(prev_shape, x)
        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

    def forward(self, x: TensorGPU) -> TensorGPU:
        return x

    def backward(self, dy: TensorGPU) -> TensorGPU:
        return dy
