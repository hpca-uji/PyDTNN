from pydtnn.activations.softmax import Softmax
from pydtnn.backends.pycuda.activations.activation import ActivationPycuda
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape

from pydtnn.libs import cudnn as cudnn
from pycuda import gpuarray  # type: ignore


class SoftmaxPycuda(Softmax[TensorGPU], ActivationPycuda):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = None
        self.algo = None

    def _model_init(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super()._model_init(prev_shape, x)

        self.mode = cudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']
        self.algo = cudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        self.memory_used += self.y.nbytes + self.dx.nbytes

    def forward(self, x: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnSoftmaxForward(self.model.cudnn_handle, self.algo, self.mode, alpha,
                                  x.desc, x.ptr, beta,
                                  self.y.desc, self.y.ptr)
        return self.y

    def backward(self, dy: TensorGPU) -> TensorGPU:
        alpha, beta = 1.0, 0.0
        cudnn.cudnnSoftmaxBackward(self.model.cudnn_handle, self.algo, self.mode, alpha,
                                   self.y.desc, self.y.ptr,
                                   dy.desc, dy.ptr, beta,
                                   self.dx.desc, self.dx.ptr)
        return self.dx
