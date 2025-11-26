import numpy as np
import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.compiler import SourceModule  #type: ignore
from pycuda.elementwise import ElementwiseKernel  #type: ignore

from pydtnn.backends.gpu.optimizers.optimizer import OptimizerGPU
from pydtnn.optimizers.nadam import Nadam

from pydtnn.backends.gpu.layers.layer import LayerGPU
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import DTYPE2CTYPE


class NadamGPU(Nadam[TensorGPU], OptimizerGPU):
    """
    NadamGPU optimizer
    """

    def __init__(self, learning_rate=1e-2, beta1=0.99, beta2=0.999, epsilon=1e-7, decay=0.0, dtype: np.dtype = np.dtype(np.float32)):
        super().__init__(learning_rate, beta1, beta2, epsilon, decay, dtype)

        func_pow = {np.dtype(np.float32): "powf", np.dtype(np.float64): "pow"}
        
        # --- GPU ---
        parameters_gpu = "{T} *w, {T} *dw, {T} *m, {T} *v, float it, " \
                         "float lr, float decay, float beta1, float beta2, float epsilon".format(T=DTYPE2CTYPE[dtype])
        operations_gpu = """
            m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
            v[i] = beta2 * v[i] + (1 - beta2) * {func}(dw[i], 2);
            w[i] -= lr * (decay * w[i] + (((m[i] + (1 - beta1) * dw[i]) / (1 - {func}(beta1, it))) / sqrtf((v[i] / (1 - {func}(beta2, it))) + epsilon)))
        """.format(func=func_pow[dtype])

        self.update_gpu = ElementwiseKernel(parameters_gpu, operations_gpu, "Nadam_kernel")
        # -----------
        
        # GPU DIRECT-
        _name = "Nadam_kernel_gpudirect"
        code = """
            __global__ void {name}({T} *w, {T} *dw, {T} *m, {T} *v,
                                   float it, float lr, float decay,
                                   float beta1, float beta2, float epsilon, int N) 
            {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < N) 
                {{
                    m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
                    v[i] = beta2 * v[i] + (1 - beta2) * {func}(dw[i], 2);
                    w[i] -= lr * (decay * w[i] + (((m[i] + (1 - beta1) * dw[i]) / (1 - {func}(beta1, it))) /
                            sqrt(v[i] / (1 - {func}(beta2, it)) + epsilon)));
                }}
            }}""".format(
                T = DTYPE2CTYPE[dtype],
                func=func_pow[dtype],
                name=_name
            )

        self.update_gpudirect = SourceModule(code).get_function(_name)
        # -----------

    def initialize(self, list_layers: list[LayerGPU]) -> None:
        for layer in list_layers:
            self.context[layer.id] = dict[str, int | gpuarray.GPUArray]()
            self.context[layer.id]["it"] = 0

            for w_ in layer.grad_vars.keys():
                w = getattr(layer, w_)
                self.context[layer.id]["m_%s" % w_] = gpuarray.zeros_like(w.ary, dtype=layer.model.dtype)
                self.context[layer.id]["v_%s" % w_] = gpuarray.zeros_like(w.ary, dtype=layer.model.dtype)

    def update(self, layer: LayerGPU) -> None:
        self.context[layer]["it"] += 1  # type: ignore (self.context[layer]["it"] is always an integer)
        it:int = self.context[layer]["it"]  # type: ignore (self.context[layer]["it"] is always an integer)

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = self.context[layer.id]["m_%s" % w_]
            v = self.context[layer.id]["v_%s" % w_]
            w: TensorGPU
            dw: TensorGPU
            m: gpuarray.GPUArray
            v: gpuarray.GPUArray

            if self.gpudirect:
                n = self.get_batch_size(w)
                threads, blocks = self.get_threads_and_blocks()

                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, m.gpudata, v.gpudata,
                                      np.float32(it), np.float32(self.learning_rate),
                                      np.float32(self.decay), np.float32(self.beta1),
                                      np.float32(self.beta2), np.float32(self.epsilon),
                                      np.int32(n),
                                      grid=(int(blocks), 1, 1), block=(int(threads), 1, 1),
                                      stream=layer.stream_2)
            else:
                self.update_gpu(w.ary, dw.ary, m, v,
                                np.float32(it), np.float32(self.learning_rate),
                                np.float32(self.decay), np.float32(self.beta1),
                                np.float32(self.beta2), np.float32(self.epsilon),
                                stream=layer.stream_2)
