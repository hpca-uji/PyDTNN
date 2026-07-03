"""CuPy backend abstract base module for PyDTNN."""

import functools

from cupy import RawKernel, RawModule  # type: ignore

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.utils.uses_cuda import UsesCudaCode

__all__ = ("BaseCupy",)


class BaseCupy(UsesCudaCode[RawModule, RawKernel], BaseNumpy):
    """Abstract base class for CuPy-based operations

    Integrating CUDA kernel compilation capabilities with NumPy-compatible backend logic.
    """

    _cuda_kernel = functools.partial(RawModule, backend="nvcc")
