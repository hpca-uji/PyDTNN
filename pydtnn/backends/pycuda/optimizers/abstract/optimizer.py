"""PyCUDA implementation of the base Optimizer class for PyDTNN."""

from typing import Any

from pycuda.driver import Function
from pycuda.elementwise import ElementwiseKernel

from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.utils.tensor_array import TensorArray
from pydtnn.optimizers.abstract.optimizer import Optimizer

__all__ = ("OptimizerPycuda",)


class OptimizerPycuda(Optimizer[TensorArray], BasePycuda):
    """Extends an Optimizer class with the attributes and methods required by GPU Optimizers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the PyCUDA optimizer with update kernels and GPU-direct functions."""
        super().__init__(*args, **kwargs)
        self.update_kernel: ElementwiseKernel = None
        self.update_gpudirect: Function = None
