import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.layers.input import Input
from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


class InputPycuda(Input[TensorGPU], LayerPycuda):

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU):
        super().initialize(prev_shape, x)

        y_gpu = gpuarray.zeros((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.real_memory_size += self.y.nbytes

    def forward(self, x: TensorGPU) -> TensorGPU:
        return x

    def backward(self, dy: TensorGPU) -> TensorGPU:
        return dy
