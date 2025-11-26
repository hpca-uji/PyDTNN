import numpy as np
import pycuda.gpuarray as gpuarray  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda.elementwise import ElementwiseKernel  # type: ignore

from pydtnn.backends.gpu.optimizers.optimizer import OptimizerGPU
from pydtnn.optimizers.rmsprop import RMSProp
from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import DTYPE2CTYPE


class RMSPropGPU(RMSProp[TensorGPU], OptimizerGPU):
    """
    RMSPropGPU Optimizer
    """

    def __init__(self, learning_rate=1e-2, rho=0.9, epsilon=1e-7, decay=0.0, dtype=np.dtype(np.float32)):
        super().__init__(learning_rate, rho, epsilon, decay, dtype)

        pow_func = {np.dtype(np.float32): "powf", np.dtype(np.float64): "pow"}[dtype]

        # --- GPU ---
        parameters_gpu = "{T} *w, {T} *dw, {T} *cache, float lr, float decay, float rho, float epsilon".format(T=DTYPE2CTYPE[dtype])
        operations_gpu = "cache[i] = rho * cache[i] + (1 - rho) * {func}(dw[i], 2); \
                                             w[i] -= lr * (decay * w[i] + (dw[i] / sqrtf(cache[i] + epsilon)))".format(func=pow_func)
        self.update_gpu = ElementwiseKernel(parameters_gpu, operations_gpu, "RMSProp_kernel")
        # -----------

        # GPU DIRECT -
        _name = "RMSProp_kernel_gpudirect"
        code = """
        __global__ void {name}({T} *w, {T} *dw, {T} *cache,
                                float lr, float decay, float rho, float epsilon, int N) 
        {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) {{
                    cache[i] = rho * cache[i] + (1 - rho) * {func}(dw[i], 2);
                    w[i] -= lr * (decay * w[i] + (dw[i] / sqrt(cache[i] + epsilon)));
                }}
        }}
        """.format(T=DTYPE2CTYPE[dtype],
                   func=pow_func,
                   name = _name
                   )
        self.update_gpudirect = SourceModule(code).get_function(_name)
        # -------------

    def initialize(self, list_layers: list[LayerGPU]) -> None:
        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, gpuarray.GPUArray]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer.id]["cache_%s" % w_] = gpuarray.zeros_like(w.ary, dtype=layer.model.dtype)

    def update(self, layer: LayerGPU):
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache = self.context[layer.id]["cache_%s" % w_]
            w: TensorGPU
            dw: TensorGPU
            cache: gpuarray.GPUArray

            if self.gpudirect:
                n = self.get_batch_size(w)
                threads, blocks = self.get_threads_and_blocks()

                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, cache.gpudata,
                                      np.float32(self.learning_rate),
                                      np.float32(self.decay), np.float32(self.rho),
                                      np.float32(self.epsilon), np.int32(n),
                                      grid=(int(blocks), 1, 1), block=(int(threads), 1, 1),
                                      stream=layer.stream_2)
            else:
                self.update_gpu(w.ary, dw.ary, cache, np.float32(self.learning_rate),
                                np.float32(self.decay), np.float32(self.rho),
                                np.float32(self.epsilon), stream=layer.stream_2)
