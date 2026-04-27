from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.layers.layer_normalization import LayerNormalization
from pycuda.compiler import SourceModule  # type: ignore
from pycuda import gpuarray  # type: ignore
import numpy as np
import logging
logger = logging.getLogger(__name__)


class LayerNormalizationPycuda(LayerNormalization[TensorArray], LayerPycuda):
    def _model_init(self, prev_shape, x):
        super()._model_init(prev_shape, x)
        self.shape = prev_shape
        self.x = x
        self.epsilon = np.float32(self.epsilon)

        # Shape same as x input
        self.y = gpuarray.zeros(x.shape, self.model.dtype)
        self.y = TensorArray(self.y, self.model.tensor_fmt, self.model.cudnn_dtype)
        self.dx = gpuarray.zeros(x.shape, self.model.dtype)
        self.dx = TensorArray(self.dx, self.model.tensor_fmt, self.model.cudnn_dtype)

        # Shape same as x input, but batch = 1. For scaling at the end: output = scale * post_normalization + bias
        gamma_shape = (int(np.prod([x.shape[i] for i in self.axis])),)
        gamma = gpuarray.to_gpu(np.full(gamma_shape, self.gamma_init_val, self.model.dtype))
        self.gamma: TensorArray = TensorArray(gamma, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorArray.TensorType.OTHER)
        beta = gpuarray.zeros(gamma_shape, self.model.dtype)
        self.beta: TensorArray = TensorArray(beta, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorArray.TensorType.OTHER)
        dgamma = gpuarray.zeros(gamma_shape, self.model.dtype)
        self.dgamma: TensorArray = TensorArray(dgamma, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorArray.TensorType.OTHER)
        dbeta = gpuarray.zeros(gamma_shape, self.model.dtype)
        self.dbeta: TensorArray = TensorArray(dbeta, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorArray.TensorType.OTHER)

        # Shape same as x input, but last layer = 1. For mean computation across the normalization axis.
        mean_shape = (int(np.prod(x.shape) / np.prod([x.shape[i] for i in self.axis])),)  # (*x.shape[:-2], 1, 1)
        std = gpuarray.zeros(mean_shape, self.model.dtype)
        self.std: TensorArray = TensorArray(std, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorArray.TensorType.OTHER)
        out_shape = x.shape
        xn = gpuarray.zeros(out_shape, self.model.dtype)
        self.xn: TensorArray = TensorArray(xn, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorArray.TensorType.OTHER)

        self.__init_kernels_gpu__()
        self.threads = int(min(self.kernel_dim_params[0], 1024))
        self.blocks = int(max(self.kernel_dim_params[0], 1024) // self.threads + 1)

        self.threads_backward_weights = int(min(self.kernel_dim_params[1], 1024))
        self.blocks_backward_weights = int(max(self.kernel_dim_params[1], 1024) // self.threads_backward_weights + 1)

    def forward(self, x):
        self.kernel_forward(x.ary, self.y.ary,
                            self.xn.ary, self.std.ary,
                            self.gamma.ary, self.beta.ary,
                            self.epsilon,
                            *self.kernel_dim_params,
                            grid=(self.blocks, 1, 1), block=(self.threads, 1, 1),
                            stream=self.model.stream)
        return self.y

    def backward(self, dy):
        self.kernel_backward(dy.ary, self.dx.ary,
                             self.xn.ary, self.std.ary,
                             self.gamma.ary, self.epsilon,
                             *self.kernel_dim_params,
                             grid=(self.blocks, 1, 1), block=(self.threads, 1, 1),
                             stream=self.model.stream)

        self.kernel_backward_weigths(dy.ary, self.xn.ary,
                                     self.dgamma.ary, self.dbeta.ary, self.epsilon,
                                     *self.kernel_dim_params,
                                     grid=(self.blocks_backward_weights, 1, 1), block=(self.threads_backward_weights, 1, 1),
                                     stream=self.model.stream)

        # print(np.sum(self.beta.get()), np.sum(self.dbeta.get()))
        return self.dx

    def __init_kernels_gpu__(self):

        self.kernel_forward = self._fwd_kernel()
        n = np.prod([self.y.shape[i] for i in self.axis])
        self.kernel_dim_params = (np.int32(np.prod(self.y.shape) // n), np.int32(n))

        self.kernel_backward = self._bwd_kernel()
        self.kernel_backward_weigths = self._get_kernel(func_name="layer_normalization_backward_weights")
        return
