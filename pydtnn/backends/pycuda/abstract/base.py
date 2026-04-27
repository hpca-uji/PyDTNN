from pycuda.compiler import SourceModule  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.abstract.base import Base
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.utils.uses_cuda import UsesCudaCode


class BasePycuda(UsesCudaCode, Base[TensorArray]):

    def _get_module(self, code: str) -> SourceModule:
        return SourceModule(code)

    def _model_init(self) -> None:
        super()._model_init()
        self._kernel_init()

    def _kernel_init(self) -> Function:
        pass
