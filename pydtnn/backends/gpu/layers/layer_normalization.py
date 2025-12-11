import numpy as np

import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from pydtnn.layers.layer_normalization import LayerNormalization
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU


class LayerNormalizationGPU(LayerNormalization[TensorGPU], LayerGPU):
    def initialize(self, prev_shape, x):
        super().initialize(prev_shape, x)
        self.shape = prev_shape
        self.x = x
        self.epsilon = np.float32(self.epsilon)

        # Shape same as x input
        self.y = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.y = TensorGPU(self.y, self.model.tensor_fmt, self.model.cudnn_dtype)
        self.dx = gpuarray.empty(x.ary.shape, self.model.dtype)
        self.dx = TensorGPU(self.dx, self.model.tensor_fmt, self.model.cudnn_dtype)

        # Shape same as x input, but batch = 1. For scaling at the end: output = scale * post_normalization + bias
        gamma_shape = (int(np.prod([x.ary.shape[i] for i in self.axis])),)
        self.gamma = gpuarray.to_gpu(np.full(gamma_shape, self.gamma_init_val, self.model.dtype))
        self.gamma = TensorGPU(self.gamma, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorGPU.TensorTypeEnum.OTHER)
        self.beta = gpuarray.zeros(gamma_shape, self.model.dtype)
        self.beta = TensorGPU(self.beta, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorGPU.TensorTypeEnum.OTHER)
        self.dgamma = gpuarray.zeros(gamma_shape, self.model.dtype)
        self.dgamma = TensorGPU(self.dgamma, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorGPU.TensorTypeEnum.OTHER)
        self.dbeta = gpuarray.zeros(gamma_shape, self.model.dtype)
        self.dbeta = TensorGPU(self.dbeta, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorGPU.TensorTypeEnum.OTHER)

        # Shape same as x input, but last layer = 1. For mean computation across the normalization axis.
        mean_shape = (int(np.prod(x.ary.shape) / np.prod([x.ary.shape[i] for i in self.axis])),)  # (*x.ary.shape[:-2], 1, 1)
        self.std = gpuarray.empty(mean_shape, self.model.dtype)
        self.std = TensorGPU(self.std, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorGPU.TensorTypeEnum.OTHER)
        out_shape = x.ary.shape
        self.xn = gpuarray.empty(out_shape, self.model.dtype)
        self.xn = TensorGPU(self.xn, self.model.tensor_fmt, self.model.cudnn_dtype, tensor_type=TensorGPU.TensorTypeEnum.OTHER)

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

        # print(np.sum(self.beta.ary.get()), np.sum(self.dbeta.ary.get()))
        return self.dx

    def __init_kernels_gpu__(self):
        module = SourceModule("""
        __global__ void gpu_forward(T *x, T *y, T *xn, T *std, T *gamma, T *beta, float epsilon, int batch, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch) {
                int i = 0;
                T mu = 0;
                T var = 0;
                T xc = 0;
                // Mean
                for ( i = 0; i < n; i++ ) {
                    mu += x[idx * n + i] / n;
                }

                // Var
                for ( i = 0; i < n; i++ ){
                    xc = x[idx * n + i] - mu;
                    var += (xc * xc) / n;
                    xn[idx * n + i] = xc;
                }
                var = sqrtf(var + epsilon);
                std[idx] = var;
                // Normalization and Scaling
                for ( i = 0; i < n; i++ ){
                    xn[idx * n + i] /= (var + epsilon);
                    y[idx * n + i] = gamma[i] * xn[idx * n +i] + beta[i];
                }
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        self.kernel_forward = module.get_function("gpu_forward")
        n = np.prod([self.y.ary.shape[i] for i in self.axis])
        self.kernel_dim_params = (np.int32(np.prod(self.y.ary.shape) // n), np.int32(n))

        module = SourceModule("""
        __global__ void gpu_backward(T *dy, T *dx, T *xn, T *std, T *gamma, float epsilon, int batch, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < batch) {
                int i = 0;
                T mean1 = 0;
                T mean2 = 0;

                // Means
                for ( i = 0; i < n; i++ ) {
                    mean1 += gamma[i] * xn[idx * n + i] * (dy[idx * n + i] / n);
                    mean2 += gamma[i] * (dy[idx * n + i] / n);
                }

                // dx
                for ( i = 0; i < n; i++ ) {
                    dx[idx * n + i] = (dy[idx * n + i] - xn[idx * n + i] * mean1 - mean2) / (std[idx] + epsilon);
                }
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        self.kernel_backward = module.get_function("gpu_backward")

        module = SourceModule("""
        __global__ void gpu_backward_weights(T *dy, T *xn, T *dgamma, T *dbeta, float epsilon, int batch, int n)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                int i = 0;
                T mean1 = 0;
                T mean2 = 0;

                // Means
                for ( i = 0; i < batch; i++ ) {
                    mean1 += xn[i * n + idx] * (dy[i * n + idx] / batch);
                    mean2 += dy[i * n + idx] / batch;
                }
                dgamma[idx] = (fabs(mean1) < epsilon) ? 0.0 : mean1;
                dbeta[idx] = (fabs(mean2) < epsilon) ? 0.0 : mean2;
            }
            return;
        }
        """.replace("T", {np.float32: "float", np.float64: "double"}[self.model.dtype]))
        self.kernel_backward_weigths = module.get_function("gpu_backward_weights")

        return
