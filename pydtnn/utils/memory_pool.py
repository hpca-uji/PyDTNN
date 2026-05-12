"""Memory management utilities for preallocating and managing tensor buffers."""

import logging
import math

import pydtnn.libs.numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = (
    "PreallocMemory",
    "PrivateMemory",
)

logger = logging.getLogger(__name__)


class PrivateMemory:
    """Base class for managing private memory allocations."""

    def __init__(self, size: int) -> None:
        """Initialize private memory with a fixed capacity."""
        self._capacity: int = size
        self._used: int = 0

    @staticmethod
    def _total(*args: int) -> int:
        """Calculate the sum of provided memory sizes."""
        return sum(args)

    def ndarray(self, shape: ArrayShape, dtype: np.dtype, order: str = "C") -> np.ndarray:
        """Create a new numpy array."""
        return np.zeros(shape, dtype, order)  # type: ignore

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, cls, exc, tb):
        """Exit the context manager."""
        pass


class PreallocMemory(PrivateMemory):
    """Memory pool implementation using a preallocated buffer."""

    def __init__(self, size: int) -> None:
        """Initialize preallocated memory buffer."""
        super().__init__(size)
        self._stack = []
        self._buffer = np.zeros(size, dtype=np.uint8)

    @staticmethod
    def _total(*args: int) -> int:
        """Calculate the maximum of provided memory sizes."""
        return max(args)

    def _malloc(self, size: int) -> memoryview:
        """Allocate a slice of the preallocated buffer."""
        start = self._used
        end = start + size

        if end > self._capacity:
            raise RuntimeError(f"Getting too much memory. Memory to get={size}, Memory occupied={self._used}, Memory after the operation={self._capacity}")

        self._used = end
        return self._buffer[start:end]

    def _free(self, size: int) -> None:
        """Release a slice of the preallocated buffer."""
        new_offset = self._used - size

        if new_offset < 0:
            raise RuntimeError(f"Removing too much memory. {self._used=}, memory to erase={size}, {new_offset=}")

        self._used = new_offset

    def __enter__(self):
        """Enter the context manager and save current offset."""
        self._stack.append(self._used)
        return self

    def __exit__(self, cls, exc, tb):
        """Exit the context manager and restore previous offset."""
        self._used = self._stack.pop()

    def ndarray(self, shape: ArrayShape, dtype: np.dtype, order: str = "C") -> np.ndarray:
        """Create a numpy array backed by the preallocated buffer."""
        if order != "C":
            raise RuntimeError("PreallocMemory only supports C order")
        buffer = self._malloc(size=int(math.prod(shape) * np.dtype(dtype).itemsize))
        array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        array.fill(0)
        return array
