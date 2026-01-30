import numpy as np
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.optimizers.optimizer import Optimizer


class OptimizerGPU(Optimizer[TensorGPU]):
    """
    Extends an Optimizer class with the attributes and methods required by GPU Optimizers.
    """

    LIMIT_THREADS_AND_BLOCKS = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpudirect = False

    def set_gpudirect(self, gpudirect: bool):
        self.gpudirect = gpudirect

    def get_batch_size(self, w: TensorGPU) -> np.int32:
        return np.int32(w.size)
        # return np.int32(np.prod(((w.shape))))

    def get_threads_and_blocks(self):
        threads = min(self.model.real_batch_size, self.LIMIT_THREADS_AND_BLOCKS)
        blocks = max(self.model.real_batch_size, self.LIMIT_THREADS_AND_BLOCKS) // threads + 1
        return threads, blocks
