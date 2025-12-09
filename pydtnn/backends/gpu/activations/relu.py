import pycuda.gpuarray as gpuarray # type: ignore

from pydtnn.activations.relu import Relu
from pydtnn.backends.gpu.activations.activation import ActivationGPU
from pydtnn.libs import libcudnn as cudnn
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


class ReluGPU(Relu[TensorGPU], ActivationGPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_desc = None

    def initialize(self, prev_shape: ArrayShape, x: TensorGPU) -> None:
        super().initialize(prev_shape, x)

        self.act_desc = cudnn.cudnnCreateActivationDescriptor()

        mode = cudnn.cudnnActivationMode['CUDNN_ACTIVATION_RELU']
        nan = cudnn.cudnnNanPropagation['CUDNN_NOT_PROPAGATE_NAN']

        # We set the maximum value to the relu to 0, which specifies the upper bound
        relu_ceiling = 0.0
        cudnn.cudnnSetActivationDescriptor(self.act_desc, mode, nan, relu_ceiling)

        # Activations y
        y_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(y_gpu, self.model.tensor_format, self.model.cudnn_dtype)

        # Derivative dx
        dx_gpu = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)

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
