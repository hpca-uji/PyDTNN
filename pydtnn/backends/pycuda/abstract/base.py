from pydtnn.abstract.base import Base
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.utils.uses_cuda import UsesCudaCode
from pycuda.compiler import SourceModule  # type: ignore


class BasePycuda(UsesCudaCode, Base[TensorArray]):

    def _get_module(self, code: str) -> SourceModule:
        return SourceModule(code)
