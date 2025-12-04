import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.driver import Function  #type: ignore

from pydtnn.metrics.metric import Metric
from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.utils.constants import ArrayShape


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

        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block

    def __init_gpu_kernel__(self) -> Function:
        raise NotImplementedError()
