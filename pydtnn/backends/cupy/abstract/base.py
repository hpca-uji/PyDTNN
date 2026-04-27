from cupy import RawKernel, RawModule  # type: ignore

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.utils.uses_cuda import UsesCudaCode


class BaseCupy(UsesCudaCode, BaseNumpy):

    def _get_module(self, code: str) -> RawKernel:
        return RawModule(code, backend=self.cupy_cuda_compiler)
