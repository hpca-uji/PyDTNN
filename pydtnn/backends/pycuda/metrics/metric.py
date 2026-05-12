"""
PyCUDA implementation of metric calculations for the PyDTNN framework.
"""

import logging

from pycuda.driver import Function  # type: ignore

from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.metrics.metric import Metric
from pydtnn.utils.constants import DTYPE2CTYPE

__all__ = ("MetricPycuda",)

logger = logging.getLogger(__name__)


class MetricPycuda(Metric[TensorArray], BasePycuda):
    """
    Extends a Metric class with the attributes and methods required by GPU Metrics.
    """

    def __init__(self, eps=1e-8):
        """
        Initializes the PyCUDA metric with a small epsilon value for numerical stability.

        Args:
            eps (float): Small value to prevent division by zero.
        """
        super().__init__(eps)
        # NOTE: The following attributes will be initializated later.
        self.grid = None
        self.block = None

    def _model_init(self) -> None:
        """
        Initializes the CUDA grid and block dimensions from the associated model.
        """
        super()._model_init()
        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block

    def _kernel_init(self) -> Function:
        """
        Initializes the CUDA kernel by setting type definitions and retrieving the function.

        Returns:
            Function: The compiled PyCUDA kernel function.
        """
        self.defines_replaces = {'"TYPE"': DTYPE2CTYPE[self.dtype]}
        self.kernel = self._get_kernel()
