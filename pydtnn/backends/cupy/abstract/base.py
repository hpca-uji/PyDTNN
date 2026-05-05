import functools

from cupy import RawKernel, RawModule  # type: ignore

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.utils.uses_cuda import UsesCudaCode


__all__ = (
    "BaseCupy",
)


class BaseCupy(UsesCudaCode[RawModule, RawKernel], BaseNumpy):
    _cuda_kernel = functools.partial(RawModule, backend="nvcc")
