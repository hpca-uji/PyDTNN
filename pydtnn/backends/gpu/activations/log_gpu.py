import numpy as np

from pydtnn.activations.log import Log
from pydtnn.backends.gpu.activations.activation_gpu import ActivationGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.types import ArrayShape, DTYPE2CTYPE

import pycuda.gpuarray as gpuarray
from pydtnn.backends.gpu.libs import libcudnn as cudnn
from pycuda.elementwise import ElementwiseKernel


class LogGPU(ActivationGPU, Log):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = None
        self.dlog = None

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> TensorGPU:
        super().initialize(prev_shape, x)

        self.log = ElementwiseKernel(
            "T *in, T *out".replace("T", DTYPE2CTYPE[self.model.dtype]),
            "out[i] = %s(1.0 / (1.0 + %s(-in[i])));" %
            ({np.float32: "logf", np.float64: "log"}[self.model.dtype],
             {np.float32: "expf", np.float64: "exp"}[self.model.dtype]),
            "log_GPU")

        self.dlog = ElementwiseKernel(
            "T *in, T *out".replace("T", DTYPE2CTYPE[self.model.dtype]),
            "out[i] = 1.0 / (1.0 + %s(in[i]));" % {np.float32: "expf", np.float64: "exp"}[self.model.dtype],
            "dlog_GPU")

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.log(x.ary, self.y.ary, stream=self.model.stream)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        # Compute dx
        self.dlog(dy.ary, self.dx.ary, stream=self.model.stream)
        return self.dx
