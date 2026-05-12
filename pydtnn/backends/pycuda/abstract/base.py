"""
PyDTNN PyCUDA backend abstract base module.
"""

from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function, Module  # type: ignore

from pydtnn.abstract.base import Base
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.utils.uses_cuda import UsesCudaCode

__all__ = ("BasePycuda",)


class BasePycuda(UsesCudaCode[Module, Function], Base[TensorArray]):
    """
    Abstract base class for PyCUDA-based operations in PyDTNN.
    """

    _cuda_kernel = SourceModule

    def _model_init(self) -> None:
        """
        Initializes the model and associated CUDA kernels.
        """
        super()._model_init()
        self._kernel_init()

    def _kernel_init(self) -> Function:
        """
        Initializes and returns the required CUDA kernel function.
        """
        pass
