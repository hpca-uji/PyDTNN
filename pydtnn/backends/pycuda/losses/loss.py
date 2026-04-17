import logging
logger = logging.getLogger(__name__)

from pycuda import gpuarray  # type: ignore
from pycuda.driver import Function  # type: ignore

from pydtnn.backends.pycuda.abstract.base import BasePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.losses.loss import Loss
from pydtnn.model import Model
from pydtnn.utils.constants import ArrayShape


class LossPycuda(Loss[TensorArray], BasePycuda):
    """
    Extends a Loss class with the attributes and methods required by GPU Losses.
    """

    def __init__(self, eps=1e-8):
        super().__init__(eps)
        # NOTE: The following attributes will be initialized later.
        self.grid = None
        self.block = None

    def _model_init(self) -> None:
        super()._model_init()
        # NOTE: the model must be executed before this one.
        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block
        self.loss = gpuarray.zeros((self.model.batch_size,), self.model.dtype)
        dx_gpu = gpuarray.zeros(self.shape, self.model.dtype)
        self.dx = TensorArray(dx_gpu, self.model.tensor_format, self.model.cudnn_dtype)
        self.kernel = self.__init_gpu_kernel__()

        self.memory_used += self.dx.nbytes + self.loss.nbytes

    def __init_gpu_kernel__(self) -> Function:
        raise NotImplementedError()
