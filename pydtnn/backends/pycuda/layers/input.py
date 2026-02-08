import numpy as np
from pycuda import gpuarray  # type: ignore

from pydtnn.layers.input import Input
from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


class InputPycuda(Input[TensorGPU], LayerPycuda):

    def _model_init(self, prev_shape: ArrayShape, x: TensorGPU):
        super()._model_init(prev_shape, x)

        y_gpu = gpuarray.empty((self.model.batch_size, *self.shape), self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes

    def forward(self, x: TensorGPU) -> TensorGPU:
        return x

    def backward(self, dy: TensorGPU) -> TensorGPU:
        return dy
