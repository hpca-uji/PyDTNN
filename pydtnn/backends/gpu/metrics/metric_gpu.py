import pycuda.gpuarray as gpuarray
from pycuda.driver import Function

from pydtnn.metrics.metric import Metric
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from model import Model
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.types import ArrayShape


class MetricGPU(Metric[TensorGPU]):
    """
    Extends a Metric class with the attributes and methods required by GPU Metrics.
    """

    def __init__(self, shape: ArrayShape, eps=1e-8):
        super().__init__(shape, eps)
        self.cost = gpuarray.empty((self.model.batch_size,), self.model.dtype)
        self.kernel = self.__init_gpu_kernel__()

    def __init_gpu_kernel__(self) -> Function:
        raise NotImplementedError()
