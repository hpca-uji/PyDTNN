import pycuda.gpuarray as gpuarray  #type: ignore
from pycuda.driver import Function  #type: ignore

from pydtnn.backends.gpu.utils.tensor_gpu import TensorGPU
from pydtnn.losses.loss import Loss
from pydtnn.model import Model
from pydtnn.utils.constants import ArrayShape

class LossGPU(Loss[TensorGPU]):
    """
    Extends a Loss class with the attributes and methods required by GPU Losses.
    """

    def __init__(self, shape: ArrayShape, eps=1e-8):
        super().__init__(shape, eps)
        # NOTE: The following attributes will be initialized later.
        self.grid = None
        self.block = None

    def initialize(self) -> None:
        super().initialize()
        # NOTE: the model must be executed before this one.
        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block
        self.loss = gpuarray.empty((self.model.batch_size,), self.model.dtype)
        dx_gpu = gpuarray.empty(self.shape, self.model.dtype)
        self.dx = TensorGPU(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.kernel = self.__init_gpu_kernel__()

    def __init_gpu_kernel__(self) -> Function:
        raise NotImplementedError()
