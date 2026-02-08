import numpy as np
from pydtnn.backends.pycuda.utils.tensor_gpu import TensorGPU
from pydtnn.layer_base import LayerBase
from pydtnn.optimizers.optimizer import Optimizer

#  TODO: Fix this.
try:
    from pycuda.driver import Function  # type: ignore
    from pycuda.elementwise import ElementwiseKernel  # type: ignore
except:
    pass


class OptimizerPycuda(Optimizer[TensorGPU]):
    """
    Extends an Optimizer class with the attributes and methods required by GPU Optimizers.
    """

    LIMIT_THREADS_AND_BLOCKS = 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpudirect = False
        self.update_kernel: ElementwiseKernel = None # type: ignore (It will be intialized later)
        self.update_gpudirect: Function = None # type: ignore (It will be intialized later)

    def set_gpudirect(self, gpudirect: bool):
        self.gpudirect = gpudirect

    def get_batch_size(self, w: TensorGPU) -> np.int32:
        return np.int32(w.size)
        # return np.int32(np.prod(((w.shape))))

    def get_threads_and_blocks(self):
        threads = min(self.model.real_batch_size, self.LIMIT_THREADS_AND_BLOCKS)
        blocks = max(self.model.real_batch_size, self.LIMIT_THREADS_AND_BLOCKS) // threads + 1
        return threads, blocks

    def _model_init(self, list_layers: list[LayerBase[TensorGPU]]) -> None:
        super()._model_init(list_layers)
        self._kernel_init()

    def _kernel_init(self) -> Function:
        pass
