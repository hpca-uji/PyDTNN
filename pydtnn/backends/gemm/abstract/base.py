"""Abstract base class for GEMM-based backends in PyDTNN."""

from pydtnn.backends.numpy.abstract.base import BaseNumpy

"""Abstract base class for GEMM-based backends in PyDTNN."""

__all__ = ("BaseGemm",)


class BaseGemm(BaseNumpy):
    """
    Base class for General Matrix Multiplication (GEMM) backend implementations.

    Inherits from BaseNumpy to provide a foundation for backends that utilize
    optimized matrix multiplication routines.
    """

    ...
