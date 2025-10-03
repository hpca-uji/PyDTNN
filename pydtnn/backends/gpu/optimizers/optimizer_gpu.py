from abc import ABC
from typing import TypeVar

from pydtnn.optimizers import Optimizer
gpuarray_t = TypeVar("gpuarray_t")

from ..tensor_gpu import TensorGPU
from numpy import int32, prod

class OptimizerGPU(Optimizer, ABC):
    """
    Extends an Optimizer class with the attributes and methods required by GPU Optimizers.
    """

    LIMIT_THREADS_AND_BLOCKS = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpudirect = False

    def set_gpudirect(self, gpudirect:bool):
        self.gpudirect = gpudirect

    def get_batch_size(self, w: TensorGPU) -> int32:
        return int32(prod((self.num_real_batches, *(w.shape[1:]))))
    
    def get_threads_and_blocks(self):
        threads = min(self.num_real_batches, self.LIMIT_THREADS_AND_BLOCKS)
        blocks = max(self.num_real_batches, self.LIMIT_THREADS_AND_BLOCKS) // threads + 1
        return threads, blocks
