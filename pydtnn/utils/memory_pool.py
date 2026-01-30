import numpy as np
from pydtnn.utils.constants import ArrayShape


class MemoryPool(object):
    def __init__(self, size: int) -> None:
        self._total: int = size
        self._offset: int = 0
        self._buffer = memoryview(bytearray(size))

    def get_buffer(self, size: int) -> memoryview:
        start = self._offset
        end = start + size

        if end > self._total:
            raise RuntimeError(f"Getting too much memory. Memory to get={size}, Memory occupied={self._offset}, Memory after the operation={self._total}")

        self._offset = end
        return self._buffer[start:end]
    # -----

    def get_ndarray(self, shape: ArrayShape, dtype: np.dtype, order: str = "C") -> np.ndarray:
        if order != "C":
            raise RuntimeError("MemoryPool only supports C order")
        buffer = self.get_buffer(size=int(np.prod(shape) * np.dtype(dtype).itemsize))
        return np.frombuffer(buffer, dtype=dtype).reshape(shape, copy=False)
    # ---

    def free_buffer(self, size: int) -> None:
        new_offset = self._offset - size
        if new_offset < 0:
            raise RuntimeError(f"Removing too much memory. {self._offset=}, memory to erase={size}, {new_offset=}")
        self._offset = new_offset
    # ---
