import logging

import numpy as np
from pycuda import gpuarray  # type: ignore
from pycuda.elementwise import ElementwiseKernel  # type: ignore

from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.backends.pycuda.optimizers.optimizer import OptimizerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.optimizers.rmsprop import RMSProp
from pydtnn.utils.constants import DTYPE2CTYPE

__all__ = (
    "RMSPropPycuda",
)

logger = logging.getLogger(__name__)


class RMSPropPycuda(RMSProp[TensorArray], OptimizerPycuda):
    """
    RMSPropPycuda Optimizer
    """

    def __init__(self, learning_rate=1e-2, rho=0.9, epsilon=1e-7, decay=0.0):
        super().__init__(learning_rate, rho, epsilon, decay)

    def _kernel_init(self) -> None:
        pow_func = {np.dtype(np.float32): "powf", np.dtype(np.float64): "pow"}[self.model.dtype]

        # --- GPU ---
        parameters_gpu = "{T} *w, {T} *dw, {T} *cache, float lr, float decay, float rho, float epsilon".format(T=DTYPE2CTYPE[self.model.dtype])
        operations_gpu = "cache[i] = rho * cache[i] + (1 - rho) * {func}(dw[i], 2); \
                                             w[i] -= lr * (decay * w[i] + (dw[i] / sqrtf(cache[i] + epsilon)))".format(func=pow_func)
        self.update_kernel = ElementwiseKernel(parameters_gpu, operations_gpu, "RMSProp_kernel")

        # GPU DIRECT -
        self.defines_replaces: dict[str, str] = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype], "powf_or_pow": pow_func}
        self.update_gpudirect = self._get_kernel(func_name_subfix="_gpudirect")

    def _model_init(self, list_layers: list[LayerPycuda]) -> None:
        super()._model_init(list_layers)  # type: ignore (The type is correct: LayerPycuda extends LayerBase)

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, gpuarray.GPUArray]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer.id]["cache_%s" % w_] = gpuarray.zeros(w.shape, dtype=layer.model.dtype)

                    self.memory_used += self.context[layer.id]["cache_%s" % w_].nbytes  # type: ignore (They are both "gpuarray" and not "int")

    def update(self, layer: LayerPycuda):
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            cache = self.context[layer.id]["cache_%s" % w_]
            w: TensorArray
            dw: TensorArray
            cache: gpuarray.GPUArray

            if self.gpudirect:
                n = self.get_batch_size(w)
                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, cache.gpudata,
                                      np.float32(self.learning_rate),
                                      np.float32(self.decay), np.float32(self.rho),
                                      np.float32(self.epsilon), np.int32(n),
                                      self.model.cuda_grid, block=self.model.cuda_block,
                                      stream=layer.stream_2)
            else:
                self.update_kernel(w.ary, dw.ary, cache, np.float32(self.learning_rate),
                                   np.float32(self.decay), np.float32(self.rho),
                                   np.float32(self.epsilon), stream=layer.stream_2)
            self._dtoh_ary(layer=layer, w_gpu=w, w_cpu=getattr(layer, f"{w_}_cpu"))
