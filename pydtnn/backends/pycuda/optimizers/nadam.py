"""PyDTNN PyCUDA Nadam optimizer implementation."""

import logging

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.elementwise import ElementwiseKernel

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.backends.pycuda.optimizers.abstract.optimizer import OptimizerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.optimizers.nadam import Nadam
from pydtnn.utils.constants import DTYPE2CTYPE

__all__ = ("NadamPycuda",)

logger = logging.getLogger(__name__)


class NadamPycuda(Nadam[TensorArray], OptimizerPycuda):
    """Nadam optimizer implementation for PyCUDA backends."""

    def __init__(
        self,
        learning_rate: float = 1e-2,
        beta1: float = 0.99,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        decay: float = 0.0,
    ) -> None:
        """Initialize the NadamPycuda optimizer."""
        super().__init__(learning_rate, beta1, beta2, epsilon, decay)

    def _kernel_init(self) -> None:
        """Initialize CUDA kernels for weight updates."""
        func_pow = {np.dtype(np.float32): "powf", np.dtype(np.float64): "pow"}

        # --- GPU ---
        parameters_gpu = (
            "{T} *w, {T} *dw, {T} *m, {T} *v, float it, float lr, float decay, float beta1, float"
            " beta2, float epsilon".format(T=DTYPE2CTYPE[self.model.dtype])
        )
        operations_gpu = """
            m[i] = beta1 * m[i] + (1 - beta1) * dw[i];
            v[i] = beta2 * v[i] + (1 - beta2) * {func}(dw[i], 2);
            w[i] -= lr * (decay * w[i] + (
                ((m[i] + (1 - beta1) * dw[i]) / (1 - {func}(beta1, it)))
                / sqrtf((v[i] / (1 - {func}(beta2, it))) + epsilon)
            ))
        """.format(func=func_pow[self.model.dtype])

        self.update_kernel = ElementwiseKernel(parameters_gpu, operations_gpu, "Nadam_kernel")

        # GPU DIRECT-
        self.defines_replaces: dict[str, str] = {
            '"TYPE"': DTYPE2CTYPE[self.model.dtype],
            "powf_or_pow": func_pow[self.model.dtype],
        }
        self.update_gpudirect = self._get_kernel(func_name_subfix="_gpudirect")

    def _model_init(self, layers: list[LayerPycuda]) -> None:
        """Initialize optimizer state for the given layers."""
        super()._model_init(layers)  # pyright: ignore[reportArgumentType]

        for layer in layers:
            self.context[layer.id] = dict[str, int | gpuarray.GPUArray]()
            self.context[layer.id]["it"] = 0

            for w_ in layer.grad_vars.keys():
                w = getattr(layer, w_)
                self.context[layer.id]["m_%s" % w_] = gpuarray.zeros(
                    w.shape, dtype=layer.model.dtype
                )
                self.context[layer.id]["v_%s" % w_] = gpuarray.zeros(
                    w.shape, dtype=layer.model.dtype
                )

                self.memory_used += (
                    self.context[layer.id]["m_%s" % w_].nbytes  # pyright: ignore[reportAttributeAccessIssue]
                    + self.context[layer.id]["v_%s" % w_].nbytes  # pyright: ignore[reportAttributeAccessIssue]
                )

    def update(self, layer: LayerPycuda, update: bool = True, sync: bool = True) -> None:
        """Perform a single optimization step on the specified layer."""
        if not layer.grad_vars or not update:
            return

        self.context[layer.id]["it"] += 1
        it: int = self.context[layer.id]["it"]  # pyright: ignore[reportAssignmentType]

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            m = self.context[layer.id]["m_%s" % w_]
            v = self.context[layer.id]["v_%s" % w_]
            w: TensorArray
            dw: TensorArray
            m: gpuarray.GPUArray
            v: gpuarray.GPUArray

            if self.use_gpudirect:
                n = self.get_batch_size(w)
                self.update_gpudirect(
                    w.ary.gpudata,
                    dw.ptr_intp,
                    m.gpudata,
                    v.gpudata,
                    np.float32(it),
                    np.float32(self.learning_rate),
                    np.float32(self.decay),
                    np.float32(self.beta1),
                    np.float32(self.beta2),
                    np.float32(self.epsilon),
                    np.int32(n),
                    self.model.cuda_grid,
                    block=self.model.cuda_block,
                    stream=layer.stream_2,
                )
            else:
                self.update_kernel(
                    w.ary,
                    dw.ary,
                    m,
                    v,
                    np.float32(it),
                    np.float32(self.learning_rate),
                    np.float32(self.decay),
                    np.float32(self.beta1),
                    np.float32(self.beta2),
                    np.float32(self.epsilon),
                    stream=layer.stream_2,
                )
            self._dtoh_ary(layer=layer, w_gpu=w, w_cpu=getattr(layer, f"{w_}_cpu"))
