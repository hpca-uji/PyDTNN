import numpy as np

from pydtnn.activations.log import Log
from pydtnn.backends.pycuda.activations.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE

from pydtnn.libs import cudnn as cudnn
from pycuda import gpuarray  # type: ignore
from pycuda.elementwise import ElementwiseKernel  # type: ignore


class LogPycuda(Log[TensorGPU], ActivationPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log: ElementwiseKernel = None
        self.dlog: ElementwiseKernel = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super()._model_init(prev_shape, x)

        self.log = ElementwiseKernel(
            "{T} *in, {T} *out".format(T=DTYPE2CTYPE[self.model.dtype]),
            "out[i] = {func_log}(1.0 / (1.0 + {func_exp}(-in[i])));".format(
                func_log={np.dtype(np.float32): "logf", np.dtype(np.float64): "log"}[self.model.dtype],
                func_exp={np.dtype(np.float32): "expf", np.dtype(np.float64): "exp"}[self.model.dtype]),
            "log_Pycuda")

        self.dlog = ElementwiseKernel(
            "{T} *in, {T} *out".format(T=DTYPE2CTYPE[self.model.dtype]),
            "out[i] = 1.0 / (1.0 + {func}(in[i]));".format(func={np.dtype(np.float32): "expf", np.dtype(np.float64): "exp"}[self.model.dtype]),
            "dlog_Pycuda")

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorGPU) -> TensorGPU:
        self.log(x.ary, self.y.ary, stream=self.model.stream)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        # Compute dx
        self.dlog(dy.ary, self.dx.ary, stream=self.model.stream)
        return self.dx
