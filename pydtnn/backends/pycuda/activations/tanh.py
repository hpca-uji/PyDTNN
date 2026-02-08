from pydtnn.activations.tanh import Tanh
from pydtnn.backends.pycuda.activations.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape

from pydtnn.libs import cudnn as cudnn
from pycuda import gpuarray  # type: ignore


class TanhPycuda(Tanh[TensorGPU], ActivationPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_desc = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super()._model_init(prev_shape, x)

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']
        nan = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, 0.0)

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationForward(self.model.cudnn_handle, self.act_desc, alpha,
                                     x.desc, x.ptr, beta,
                                     self.y.desc, self.y.ptr)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationBackward(self.model.cudnn_handle, self.act_desc, alpha,
                                      self.y.desc, self.y.ptr,
                                      dy.desc, dy.ptr,
                                      self.x.desc, self.x.ptr, beta,
                                      self.dx.desc, self.dx.ptr)
        return self.dx
