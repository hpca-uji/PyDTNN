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
    def __init__(self, size: int) -> None:
        self._capacity: int = size
        self._used: int = 0

    @staticmethod
    def _total(*args: int) -> int:
        return sum(args)

    def ndarray(self, shape: ArrayShape, dtype: np.dtype, order: str = "C") -> np.ndarray:
        return np.zeros(shape, dtype, order)  # type: ignore

    def __enter__(self):
        return self

    def __exit__(self, cls, exc, tb):
        pass


class PreallocMemory(PrivateMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self._stack = []
        self._buffer = np.zeros(size, dtype=np.uint8)

    @staticmethod
    def _total(*args: int) -> int:
        return max(args)

    def _malloc(self, size: int) -> memoryview:
        start = self._used
        end = start + size

        if end > self._capacity:
            raise RuntimeError(f"Getting too much memory. Memory to get={size}, Memory occupied={self._used}, Memory after the operation={self._capacity}")

        self._used = end
        return self._buffer[start:end]

    def _free(self, size: int) -> None:
        new_offset = self._used - size

        if new_offset < 0:
            raise RuntimeError(f"Removing too much memory. {self._used=}, memory to erase={size}, {new_offset=}")

        self._used = new_offset

    def __enter__(self):
        self._stack.append(self._used)
        return self

    def __exit__(self, cls, exc, tb):
        self._used = self._stack.pop()

    def ndarray(self, shape: ArrayShape, dtype: np.dtype, order: str = "C") -> np.ndarray:
        if order != "C":
            raise RuntimeError("PreallocMemory only supports C order")
        buffer = self._malloc(size=int(math.prod(shape) * np.dtype(dtype).itemsize))
        array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
        array.fill(0)
        return array
