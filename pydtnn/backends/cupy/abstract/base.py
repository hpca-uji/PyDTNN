"""CuPy backend abstract base module for PyDTNN."""

import functools
from typing import TYPE_CHECKING

import cupy as cp
from cupy import RawKernel, RawModule  # pyright: ignore[reportAttributeAccessIssue]

from pydtnn.backends.numpy.abstract.base import BaseNumpy
from pydtnn.libs import numpy as np
from pydtnn.utils.uses_cuda import UsesCudaCode

__all__ = ("BaseCupy",)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class BaseCupy(UsesCudaCode[RawModule, RawKernel], BaseNumpy):
    """Abstract base class for CuPy-based operations

    Integrating CUDA kernel compilation capabilities with NumPy-compatible backend logic.
    """

    _cuda_kernel = staticmethod(functools.partial(RawModule, backend="nvcc"))

    def _model_init(self) -> None:
        """
        Initialize the layer model parameters and verify backend compatibility.

        Args:
            prev_shape: The shape of the input data from the previous layer.
            x: Optional input data for initialization.
        """
        super()._model_init()

        if self.model.cupy_device is None:
            from pydtnn.autoinit import errors  # NOTE: differed execution
            raise ExceptionGroup("CuPy layers requires CUDA support!", errors)
        elif np.ndarray is not cp.ndarray:  # pyright: ignore[reportAttributeAccessIssue]
            raise RuntimeError("CuPy layers requires PYDTNN_CUPY enabled!")
