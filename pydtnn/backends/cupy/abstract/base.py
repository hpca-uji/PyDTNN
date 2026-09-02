"""CuPy backend abstract base module for PyDTNN."""

import functools

import cupy as cp
from cupy import RawKernel, RawModule  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.libs import numpy as libnp
from pydtnn.utils.uses_cuda import UsesCudaCode

__all__ = ("BaseCupy",)


class BaseCupy(UsesCudaCode[RawModule, RawKernel], BaseNumpy):
    """Abstract base class for CuPy-based operations

    Integrating CUDA kernel compilation capabilities with NumPy-compatible backend logic.
    """

    _cuda_kernel = functools.partial(RawModule, backend="nvcc")

    def _model_init(self) -> None:
        """
        Initialize the layer model parameters and verify backend compatibility.

        Args:
            prev_shape: The shape of the input data from the previous layer.
            x: Optional input data for initialization.
        """
        super()._model_init()

        if libnp.ndarray is not cp.ndarray:  # pyright: ignore[reportAttributeAccessIssue]
            raise RuntimeError("CuPy layers requies PYDTNN_CUPY enabled!")
