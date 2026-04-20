from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.utils.uses_cuda import UsesCudaCode
import cupy as cp


class BaseCupy(UsesCudaCode, BaseNumpy):

    def _get_module(self, code: str) -> cp.RawKernel:
        return cp.RawModule(code, backend=self.cupy_cuda_compiler)
