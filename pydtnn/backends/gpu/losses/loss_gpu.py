from abc import ABC, abstractmethod

import pycuda.gpuarray as gpuarray
from pycuda.driver import Function

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
from pydtnn.backends.gpu import TensorGPU
from pydtnn.losses import Loss
from pydtnn.utils.types import ArrayShape

class LossGPU(Loss, ABC):
    """
    Extends a Loss class with the attributes and methods required by GPU Losses.
    """

    LIMIT_THREADS_AND_BLOCKS = 1024

    def __init__(self, shape: ArrayShape, model: "Model", eps=1e-8):
        super().__init__(shape, model, eps)
        self.loss = gpuarray.empty((self.model.batch_size,), self.model.dtype)
        dx_gpu = gpuarray.empty(self.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.kernel = self.__init_gpu_kernel__()

    @abstractmethod
    def __init_gpu_kernel__(self) -> Function:
        pass

    def get_threads_and_blocks(self):
        threads = min(self.model.real_batche_size, self.LIMIT_THREADS_AND_BLOCKS)
        blocks = max(self.model.real_batche_size, self.LIMIT_THREADS_AND_BLOCKS) // threads + 1
        return threads, blocks
