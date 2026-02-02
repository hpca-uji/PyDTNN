import pydtnn.libs.numpy as np
from pydtnn.utils.constants import ArrayShape


class PrivateMemory:
    def __init__(self, size: int) -> None:
        self._total: int = size
        self._used: int = 0

    def _malloc(self, size: int) -> memoryview:
        return memoryview(np.zeros(size, dtype=np.uint8))
    # -----

    def ndarray(self, shape: ArrayShape, dtype: np.dtype, order: str = "C") -> np.ndarray:
        return np.zeros(shape, dtype, order)  # type: ignore
    # ---

    def _free(self, size: int) -> None:
        pass


class PreallocMemory(PrivateMemory):
    def __init__(self, size: int) -> None:
        super().__init__(size)
        self._buffer = memoryview(np.zeros(size, dtype=np.uint8))

    def _malloc(self, size: int) -> memoryview:
        start = self._used
        end = start + size

        if end > self._total:
            raise RuntimeError(f"Getting too much memory. Memory to get={size}, Memory occupied={self._used}, Memory after the operation={self._total}")

        self._used = end
        return self._buffer[start:end]
    # -----

    def ndarray(self, shape: ArrayShape, dtype: np.dtype, order: str = "C") -> np.ndarray:
        if order != "C":
            raise RuntimeError("PreallocMemory only supports C order")
        buffer = self._malloc(size=int(np.prod(shape) * np.dtype(dtype).itemsize))
        return np.frombuffer(buffer, dtype=dtype).reshape(shape, copy=False)
    # ---

    def _free(self, size: int) -> None:
        new_offset = self._used - size
        if new_offset < 0:
            raise RuntimeError(f"Removing too much memory. {self._used=}, memory to erase={size}, {new_offset=}")
        self._used = new_offset
    # ---
