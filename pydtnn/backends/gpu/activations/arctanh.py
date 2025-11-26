import numpy as np

from pydtnn.activations.arctanh import Arctanh
from pydtnn.backends.gpu.activations.activation import ActivationGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU

import pycuda.gpuarray as gpuarray # type: ignore
from pycuda.elementwise import ElementwiseKernel # type: ignore
from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE


class ArctanhGPU(Arctanh[TensorGPU], ActivationGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atanh: ElementwiseKernel = None
        self.datanh: ElementwiseKernel = None

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        self.atanh = ElementwiseKernel(
            "{T} *in, {T} *out".format(T=DTYPE2CTYPE[self.model.dtype]),
            "out[i] = {func}(in[i]);" .format(func={np.float32: "atanhf", np.float64: "atanh"}[self.model.dtype]),
            "atanh")

        self.datanh = ElementwiseKernel(
            "{T} *in, {T} *out".format("T", DTYPE2CTYPE[self.model.dtype]),
            "out[i] = 1.0 / (1.0 + {func}(in[i], 2));".format(func={np.float32: "powf", np.float64: "pow"}[self.model.dtype]),
            "datanh")

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.atanh(x.ary, self.y.ary, stream=self.model.stream)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        # Compute dx
        self.datanh(dy.ary, self.dx.ary, stream=self.model.stream)
        return self.dx
