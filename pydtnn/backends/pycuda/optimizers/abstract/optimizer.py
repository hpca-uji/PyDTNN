"""PyCUDA implementation of the base Optimizer class for PyDTNN."""

import logging
from typing import Any

import numpy as np
from pycuda.driver import Function  # type: ignore
from pycuda.elementwise import ElementwiseKernel  # type: ignore

from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.optimizers.abstract.optimizer import Optimizer

__all__ = ("OptimizerPycuda",)

logger = logging.getLogger(__name__)


class OptimizerPycuda(Optimizer[TensorArray], BasePycuda):
    """Extends an Optimizer class with the attributes and methods required by GPU Optimizers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the PyCUDA optimizer with update kernels and GPU-direct functions."""
        super().__init__(*args, **kwargs)
        self.update_kernel: ElementwiseKernel = None  # type: ignore (It will be intialized later)
        self.update_gpudirect: Function = None  # type: ignore (It will be intialized later)

    def get_batch_size(self, w: TensorArray) -> np.int32:
        """
        Calculates the batch size based on the total number of elements in the tensor.

        Args:
            w: The input TensorArray.

        Returns:
            The size of the tensor as a 32-bit integer.
        """
        return np.int32(w.size)
        # return np.int32(np.prod(((w.shape))))

    def _dtoh_ary(self, layer: Layerable, w_gpu: TensorArray, w_cpu: np.ndarray) -> None:
        """
        Performs an asynchronous device-to-host transfer of tensor data.

        Args:
            layer: The layer associated with the tensor.
            w_gpu: The source TensorArray on the GPU.
            w_cpu: The destination numpy array on the host.
        """
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            # self.model.stream.synchronize()
            w_gpu.get_async(layer.stream_2, w_cpu)
