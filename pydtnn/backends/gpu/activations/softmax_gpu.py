from pydtnn.activations import Softmax
from pydtnn.backends.gpu.activations.activation_gpu import ActivationGPU
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.types import ArrayShape

import pycuda.gpuarray as gpuarray
from pydtnn.backends.gpu.libs import libcudnn as cudnn


class SoftmaxGPU(ActivationGPU, Softmax):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = None
        self.algo = None

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        self.mode = cudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']
        self.algo = cudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

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
