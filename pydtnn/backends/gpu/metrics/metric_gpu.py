from abc import ABC, abstractmethod

# noinspection PyUnresolvedReferences
import pycuda.gpuarray as gpuarray
# noinspection PyUnresolvedReferences
from pycuda.driver import Function

from pydtnn.metrics import Metric
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model import Model
from pydtnn.backends.gpu import TensorGPU

class MetricGPU(Metric[TensorGPU], ABC):
    """
    Extends a Metric class with the attributes and methods required by GPU Metrics.
    """

    def __init__(self, shape: tuple[int, ...], model: "Model", eps=1e-8):
        super().__init__(shape, model, eps)
        self.cost = gpuarray.empty((self.model.batch_size,), self.model.dtype)
        self.kernel = self.__init_gpu_kernel__()

    @abstractmethod
    def __init_gpu_kernel__(self) -> Function:
        pass
