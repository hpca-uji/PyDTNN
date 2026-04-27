from pydtnn.utils.constants import ArrayShape, DTYPE2CTYPE
from pycuda.elementwise import ElementwiseKernel  # type: ignore
from pycuda import gpuarray  # type: ignore
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.activations.activation import ActivationPycuda
from pydtnn.activations.arctanh import Arctanh
import numpy as np
import logging
logger = logging.getLogger(__name__)


class ArctanhPycuda(Arctanh[TensorArray], ActivationPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atanh: ElementwiseKernel = None
        self.datanh: ElementwiseKernel = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        self.atanh = ElementwiseKernel(
            "{T} *in, {T} *out".format(T=DTYPE2CTYPE[self.model.dtype]),
            "out[i] = {func}(in[i]);".format(func={np.dtype(np.float32): "atanhf", np.dtype(np.float64): "atanh"}[self.model.dtype]),
            "k_atanh")

        self.datanh = ElementwiseKernel(
            "{T} *in, {T} *out".format(T=DTYPE2CTYPE[self.model.dtype]),
            "out[i] = 1.0 / (1.0 + {func}(in[i], 2));".format(func={np.dtype(np.float32): "powf", np.dtype(np.float64): "pow"}[self.model.dtype]),
            "datanh")

        # Activations y
        y_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        self.atanh(x.ary, self.y, stream=self.model.stream)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        # Compute dx
        self.datanh(dy.ary, self.dx.ary, stream=self.model.stream)
        return self.dx
