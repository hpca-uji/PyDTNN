"""PyDTNN PyCUDA backend abstract base module."""

from __future__ import annotations

import functools

from pycuda.compiler import SourceModule
from pycuda.driver import Function, Module

from pydtnn.abstract.base import Base
from pydtnn.utils.tensor_array import TensorArray
from pydtnn.utils.uses_cuda import UsesCudaCode

__all__ = ("BasePycuda",)


class BasePycuda(UsesCudaCode[Module, Function], Base[TensorArray]):
    """Abstract base class for PyCUDA-based operations in PyDTNN."""

    _cuda_kernel = staticmethod(functools.partial(SourceModule, nvcc="nvcc"))

    def _model_init(self) -> None:
        """Initializes the model and associated CUDA kernels."""
        super()._model_init()
        if self.model.pycuda_device is None or self.model.cublas_handle is None or self.model.cudnn_handle is None:
            from pydtnn.autoinit import errors  # NOTE: differed execution
            raise ExceptionGroup("PyCUDA layers requires CUDA support!", errors)
        self._kernel_init()

    def _kernel_init(self) -> Function:
        """Initializes and returns the required CUDA kernel function."""
