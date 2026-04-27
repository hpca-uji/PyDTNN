from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.layers.layer import LayerPycuda
from pydtnn.optimizers.sgd import SGD
from pydtnn.backends.pycuda.optimizers.optimizer import OptimizerPycuda
from pycuda.elementwise import ElementwiseKernel  # type: ignore
from pycuda.compiler import SourceModule  # type: ignore
from pycuda import gpuarray  # type: ignore
import numpy as np
import logging
logger = logging.getLogger(__name__)


class SGDPycuda(SGD[TensorArray], OptimizerPycuda):
    """
    SGDPycuda optimizer
    """

    def __init__(self, learning_rate=1e-2, momentum=0.9, nesterov=False, decay=0.0):
        super().__init__(learning_rate, momentum, nesterov, decay)

    def _kernel_init(self) -> None:
        # --- GPU ---
        parameters_gpu = "{T} *w, {T} * dw, {T} * v, float lr, float decay, float momentum".format(T=DTYPE2CTYPE[self.model.dtype])
        ops_gpu = {True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
                   False: "w[i] -= lr * (decay * w[i] + v[i])"}[self.nesterov]
        operations_gpu = "v[i] = momentum * v[i] + dw[i]; {nesterov_ops};".format(nesterov_ops=ops_gpu)

        self.update_kernel = ElementwiseKernel(parameters_gpu, operations_gpu, "SGD_kernel")
        # ------------

        # GPU Direct -
        self.defines_replaces: dict[str, str] = {"\"TYPE\"": DTYPE2CTYPE[self.model.dtype],
                                                 "NESTEROV_OPS": "NESTEROV_OPS" if self.nesterov else "NOT_NESTEROV"}
        self.update_gpudirect = self._get_kernel(func_name_subfix="_gpudirect")
        # ------------

    def _model_init(self, list_layers: list[LayerPycuda]) -> None:
        super()._model_init(list_layers)  # type: ignore (The type is correct: LayerPycuda extends LayerBase)

        for layer in list_layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, gpuarray.GPUArray]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer.id]["velocity_%s" % w_] = gpuarray.zeros_like(w.ary, dtype=w.ary.dtype)

                    self.memory_used += self.context[layer.id]["velocity_%s" % w_].nbytes  # type: ignore (They are both "gpuarray" and not "int")

    def update(self, layer: LayerPycuda):
        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity = self.context[layer.id]["velocity_%s" % w_]
            w: TensorArray
            dw: TensorArray
            velocity: gpuarray.GPUArray

            if self.gpudirect:
                n = self.get_batch_size(w)
                self.update_gpudirect(w.ary.gpudata, dw.ptr_intp, velocity.gpudata,
                                      np.float32(self.learning_rate), np.float32(self.decay),
                                      np.float32(self.momentum), np.int32(n),
                                      self.model.cuda_grid, block=self.model.cuda_block,
                                      stream=layer.stream_2)
            else:
                n = np.int32(np.prod(w.shape))
                self.update_kernel(w.ary, dw.ary, velocity, np.float32(self.learning_rate),
                                   np.float32(self.decay), np.float32(self.momentum),
                                   stream=layer.stream_2)
            self._dtoh_ary(layer=layer, w_gpu=w, w_cpu=getattr(layer, f"{w_}_cpu"))
