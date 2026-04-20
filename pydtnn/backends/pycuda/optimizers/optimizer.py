from pycuda.elementwise import ElementwiseKernel  # type: ignore
from pycuda.driver import Function  # type: ignore
from pydtnn.optimizers.optimizer import Optimizer
from pydtnn.abstract.layerable import Layerable
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.backends.pycuda.abstract.base import BasePycuda
import numpy as np
import logging
logger = logging.getLogger(__name__)
from pydtnn.utils.uses_cuda import PyCudaCudaCode


class OptimizerPycuda(Optimizer[TensorArray], BasePycuda, PyCudaCudaCode):

    """
    Extends an Optimizer class with the attributes and methods required by GPU Optimizers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_kernel: ElementwiseKernel = None  # type: ignore (It will be intialized later)
        self.update_gpudirect: Function = None  # type: ignore (It will be intialized later)

    def get_batch_size(self, w: TensorArray) -> np.int32:
        return np.int32(w.size)
        # return np.int32(np.prod(((w.shape))))

    def _model_init(self, list_layers: list[Layerable[TensorArray]]) -> None:
        super()._model_init(list_layers)
        self._kernel_init()

    def _kernel_init(self) -> Function:
        pass

    def _dtoh_ary(self, layer: Layerable, w_gpu: TensorArray, w_cpu: np.ndarray) -> None:
        if self.model.comm and not self.model.gpudirect and not self.model.enable_nccl:
            # self.model.stream.synchronize()
            w_gpu.ary.get_async(layer.stream_2, w_cpu)
