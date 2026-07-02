"""PyCUDA backend implementation for loss functions."""

import logging

from pycuda import gpuarray  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.abstract.loss import Loss
from pydtnn.utils.constants import DTYPE2CTYPE

__all__ = ("LossPycuda",)

logger = logging.getLogger(__name__)


class LossPycuda(Loss[TensorArray], BasePycuda):
    """Extends a Loss class with the attributes and methods required by GPU Losses."""

    def __init__(self, eps: float = 1e-8) -> None:
        """
        Initializes the PyCUDA loss base class.

        Args:
            weights (list[float] | None): The list of the weight of every class. 
             If it's None, all classes will have the same weight. default: None.
            eps (float): Small value to prevent division by zero.
        """
        super().__init__(eps)
        # NOTE: The following attributes will be initialized later.
        self.grid = None
        self.block = None

    def _weights_to_tensor(self, weights: list[float] | None) -> TensorArray:
        w = super()._weights_to_tensor(weights)
        w = TensorArray.to_gpu(ary=w,
                               tensor_format=self.model.tensor_format,
                               cudnn_dtype=self.model.cudnn_dtype)
        return w

    def _model_init(self) -> None:
        """Initializes GPU memory buffers and model-dependent parameters."""
        super()._model_init()
        # NOTE: the model must be executed before this one.
        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block
        self.loss = gpuarray.zeros((self.model.batch_size,), self.model.dtype)
        dx_gpu = gpuarray.zeros(self.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.memory_used += self.dx.nbytes + self.loss.nbytes

    def _kernel_init(self) -> Function:
        """
        Prepares kernel definitions and retrieves the compiled CUDA function.

        Returns:
            Function: The compiled PyCUDA kernel function.
        """
        self.defines_replaces = {'"TYPE"': DTYPE2CTYPE[self.model.dtype]}
        self.kernel = self._get_kernel()
