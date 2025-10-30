import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.driver import Function  #type: ignore

from pydtnn.metrics.metric import Metric
from pydtnn.backends.gpu.tensor_gpu import TensorGPU
from pydtnn.utils.types import ArrayShape


class MetricGPU(Metric[TensorGPU]):
    """
    Extends a Metric class with the attributes and methods required by GPU Metrics.
    """

    def __init__(self, shape: ArrayShape, eps=1e-8):
        super().__init__(shape, eps)
    
    def initialize(self) -> None:
        super().initialize()
        self.cost = gpuarray.empty((self.model.batch_size,), self.model.dtype)
        self.kernel = self.__init_gpu_kernel__()
        
        self.threads = min(self.model.batch_size, 1024)
        self.blocks = max(self.model.batch_size, 1024) // self.threads + 1
        
        self.grid = (self.blocks, 1, 1)
        self.block = (self.threads, 1, 1)

    def __init_gpu_kernel__(self) -> Function:
        raise NotImplementedError()
