import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel

from pydtnn.backends.gpu.optimizers.optimizer_gpu import OptimizerGPU, gpuarray_t
from pydtnn.optimizers import SGD
from pydtnn.backends.gpu.layers import LayerGPU
from pydtnn.backends.gpu import TensorGPU


class SGDGPU(OptimizerGPU, SGD):
    """
    SGDGPU optimizer
    """

    def __init__(self, learning_rate=1e-2, momentum=0.9, nesterov=False, decay=0.0, dtype: np.dtype = np.float32):
        super().__init__(learning_rate, momentum, nesterov, decay, dtype)

        if not self.gpudirect:
            self.update_gpu = ElementwiseKernel("T *w, T * dw, T * v, \
                               float lr, float decay, float momentum".replace("T",
                                                                              {np.float32: "float",
                                                                               np.float64: "double"}[dtype]),
                                                "v[i] = momentum * v[i] + dw[i]; %s;" %
                                                ({True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
                                                  False: "w[i] -= lr * (decay * w[i] + v[i])"}[nesterov]),
                                                "SGD_kernel")
        else:
            self.update_gpu = SourceModule(("""
            __global__ void SGD_kernel(T *w, T *dw, T *v,
                               float lr, float decay, float momentum, int N) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {
                    v[i] = momentum * v[i] + dw[i];
                    %s;
                }
             }""" % ({True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
                      False: "w[i] -= lr * (decay * w[i] + v[i])"}[nesterov])).replace("T",
                                                                                       {np.float32: "float",
                                                                                        np.float64: "double"}[dtype])
            ).get_function("SGD_kernel")
    
    def initialize(self, list_layers: list[LayerGPU]) -> None:
        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, gpuarray_t]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer.id]["velocity_%s" % w_] = gpuarray.zeros_like(w.ary, dtype=w.ary.dtype)

    def update(self, layer: LayerGPU):
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity = self.context[layer.id]["velocity_%s" % w_]
            w: TensorGPU
            dw: TensorGPU
            velocity: gpuarray

            if self.gpudirect:
                n = self.get_batch_size(w)
                threads, blocks = self.get_threads_and_blocks()

                self.update_gpu(w.ary.gpudata, dw.ptr_intp, velocity.gpudata,
                                np.float32(self.learning_rate), np.float32(self.decay),
                                np.float32(self.momentum), np.int32(n),
                                grid=(int(blocks), 1, 1), block=(int(threads), 1, 1),
                                stream=layer.stream_2)
            else:
                n = np.int32(np.prod(w.shape))
                self.update_gpu(w.ary, dw.ary, velocity, np.float32(self.learning_rate),
                                np.float32(self.decay), np.float32(self.momentum),
                                stream=layer.stream_2)
