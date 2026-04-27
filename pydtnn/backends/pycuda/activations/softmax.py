from pycuda import gpuarray  # type: ignore
from pydtnn.libs import cudnn as cudnn
from pydtnn.utils.constants import ArrayShape
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.activations.activation import ActivationPycuda
from pydtnn.activations.softmax import Softmax
import logging
logger = logging.getLogger(__name__)


class SoftmaxPycuda(Softmax[TensorArray], ActivationPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = None
        self.algo = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray) -> None:
        super()._model_init(prev_shape, x)

        self.mode = cudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']
        self.algo = cudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']

        # Activations y
        y_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.zeros(x.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorArray) -> TensorArray:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnSoftmaxForward(self.model.cudnn_handle, self.algo, self.mode, alpha,
                                  x.desc, x.ptr_voidp, beta,
                                  self.y.desc, self.y.ptr_voidp)
        return self.y

    def backward(self, dy: TensorArray) -> TensorArray:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnSoftmaxBackward(self.model.cudnn_handle, self.algo, self.mode, alpha,
                                   self.y.desc, self.y.ptr_voidp,
                                   dy.desc, dy.ptr_voidp, beta,
                                   self.dx.desc, self.dx.ptr_voidp)
        return self.dx
