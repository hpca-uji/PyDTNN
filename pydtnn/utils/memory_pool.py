import numpy as np
from pydtnn.utils.constants import ArrayShape

class Memory_Pool(object):

    _instance: "Memory_Pool" = None  #type: ignore (Only is none before it's intialization)

    @staticmethod
    def instance(cls: "Memory_Pool"):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)

        return cls._instance

    def __init__(self, size: int, dtype: np.dtype) -> None:
        self.total_memory: int = size
        self.off_set_free:int = 0
        self.memory_pool: np.ndarray = np.zeros(self.total_memory, dtype=dtype, order="C")

    def get_memory(self, size:int) -> np.ndarray:
        start = self.off_set_free
        end = start + size

        if end > self.total_memory:
            raise RuntimeError(f"Getting too much memory. Memory to get= {size}, Memory occupied= {self.off_set_free}, Memory after the operation= {self.total_memory}")

        self.off_set_free = end
        print(f" {start=}, {end=} [{self.off_set_free}/{self.total_memory}] || {self.off_set_free=} {self.total_memory=}") # TODO: BORRAR
        return self.memory_pool[start:end]
    # -----

    def get_ndarray(self, shape:ArrayShape | tuple[int, ...]) -> np.ndarray:
        new_array = self.get_memory(size=int(np.prod(shape)))
        return new_array.reshape(shape, copy=False)
    # ---

    def free_memory(self, size:int) -> None:
        new_offset = self.off_set_free - size
        if new_offset < 0:
            raise RuntimeError(f"Removing too much memory. {self.off_set_free=}, memory to erase={size}, {new_offset=}")
        self.off_set_free = new_offset

        print(f" {self.off_set_free}/{self.total_memory} || {self.off_set_free=} {self.total_memory=}") # TODO: BORRAR
    # ---
