from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function, Module  # type: ignore

from pydtnn.abstract.base import Base
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.utils.uses_cuda import UsesCudaCode

__all__ = ("BasePycuda",)


class BasePycuda(UsesCudaCode[Module, Function], Base[TensorArray]):
    _cuda_kernel = SourceModule

    def _model_init(self) -> None:
        super()._model_init()
        self._kernel_init()

    def _kernel_init(self) -> Function:
        pass
