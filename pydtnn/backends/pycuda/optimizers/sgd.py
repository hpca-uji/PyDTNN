"""PyCUDA implementation of the Stochastic Gradient Descent (SGD) optimizer."""

import logging

import numpy as np
from pycuda import gpuarray  # pyright: ignore[reportAttributeAccessIssue]
from pycuda.elementwise import ElementwiseKernel

from pydtnn.backends.pycuda.layers.abstract.layer import LayerPycuda
from pydtnn.backends.pycuda.optimizers.abstract.optimizer import OptimizerPycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.optimizers.sgd import SGD
from pydtnn.utils.constants import DTYPE2CTYPE

__all__ = ("SGDPycuda",)

logger = logging.getLogger(__name__)


class SGDPycuda(SGD[TensorArray], OptimizerPycuda):
    """PyCUDA-accelerated Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        learning_rate: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = False,
        decay: float = 0.0,
    ) -> None:
        """
        Initializes the SGDPycuda optimizer.

        Args:
            learning_rate (float): Step size for parameter updates.
            momentum (float): Momentum factor.
            nesterov (bool): Whether to use Nesterov momentum.
            decay (float): Weight decay factor.
        """
        super().__init__(learning_rate, momentum, nesterov, decay)

    def _kernel_init(self) -> None:
        """Initializes the PyCUDA elementwise kernels for parameter updates."""
        # --- GPU ---
        parameters_gpu = "{T} *w, {T} * dw, {T} * v, float lr, float decay, float momentum".format(
            T=DTYPE2CTYPE[self.model.dtype]
        )
        ops_gpu = {
            True: "w[i] -= lr * (decay * w[i] + dw[i] + momentum * v[i])",
            False: "w[i] -= lr * (decay * w[i] + v[i])",
        }[self.nesterov]
        operations_gpu = "v[i] = momentum * v[i] + dw[i]; {nesterov_ops};".format(
            nesterov_ops=ops_gpu
        )

        self.update_kernel = ElementwiseKernel(parameters_gpu, operations_gpu, "SGD_kernel")

        # GPU Direct -
        self.defines_replaces: dict[str, str] = {
            '"TYPE"': DTYPE2CTYPE[self.model.dtype],
            "NESTEROV_OPS": "NESTEROV_OPS" if self.nesterov else "NOT_NESTEROV",
        }
        self.update_gpudirect = self._get_kernel(func_name_subfix="_gpudirect")

    def _model_init(self, layers: list[LayerPycuda]) -> None:
        """
        Initializes optimizer state (velocity buffers) for each layer.

        Args:
            layers (list[LayerPycuda]): List of layers to track.
        """
        # NOTE: The type is correct: LayerPycuda extends LayerBase
        super()._model_init(layers)  # pyright: ignore[reportArgumentType]

        for layer in layers:
            list_grad_vars = list(layer.grad_vars.keys())

            if len(list_grad_vars) != 0:
                self.context[layer.id] = dict[str, gpuarray.GPUArray]()
                for w_ in list_grad_vars:
                    w = getattr(layer, w_)
                    self.context[layer.id]["velocity_%s" % w_] = gpuarray.zeros(
                        w.shape, dtype=w.dtype
                    )

                    # NOTE: They are both "gpuarray" and not "int"
                    self.memory_used += self.context[layer.id]["velocity_%s" % w_].nbytes  # pyright: ignore[reportAttributeAccessIssue] # noqa: E501

    def update(self, layer: LayerPycuda, update: bool = True, sync: bool = True) -> None:
        """
        Performs a single optimization step on the provided layer.

        Args:
            layer (LayerPycuda): The layer to update.
        """
        if not layer.grad_vars or not update:
            return

        for w_, dw_ in layer.grad_vars.items():
            w, dw = getattr(layer, w_), getattr(layer, dw_)
            velocity = self.context[layer.id]["velocity_%s" % w_]
            w: TensorArray
            dw: TensorArray
            velocity: gpuarray.GPUArray

            if self.model.use_gpudirect:
                n = self.get_batch_size(w)
                self.update_gpudirect(
                    w.gpudata,
                    dw.ptr_intp,
                    velocity.gpudata,
                    np.float32(self.learning_rate),
                    np.float32(self.decay),
                    np.float32(self.momentum),
                    np.int32(n),
                    self.model.cuda_grid,
                    block=self.model.cuda_block,
                    stream=layer.stream_2,
                )
            else:
                n = np.int32(np.prod(w.shape))
                self.update_kernel(
                    w.ary,
                    dw.ary,
                    velocity,
                    np.float32(self.learning_rate),
                    np.float32(self.decay),
                    np.float32(self.momentum),
                    stream=layer.stream_2,
                )
            self._dtoh_ary(layer=layer, w_gpu=w, w_cpu=getattr(layer, f"{w_}_cpu"))
