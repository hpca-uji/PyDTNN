import logging

from pycuda import gpuarray  # type:ignore

from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.backends.pycuda.activations.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.libs import cudnn as cudnn
from pydtnn.utils.constants import ArrayShape

__all__ = (
    "SigmoidPycuda",
)

logger = logging.getLogger(__name__)


class SigmoidPycuda(Sigmoid[TensorArray], ActivationPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_desc = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode['CUDNN_ACTIVATION_SIGMOID']
        nan = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, 0.0)

        # Activations a
        y_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationForward(self.model.cudnn_handle, self.act_desc, alpha,
                                     x.desc, x.ptr_voidp, beta,
                                     self.y.desc, self.y.ptr_voidp)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnActivationBackward(self.model.cudnn_handle, self.act_desc, alpha,
                                      self.y.desc, self.y.ptr_voidp,
                                      dy.desc, dy.ptr_voidp,
                                      self.x.desc, self.x.ptr_voidp, beta,
                                      self.dx.desc, self.dx.ptr_voidp)
        return self.dx
