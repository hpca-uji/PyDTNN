import logging

from pydtnn.activations.activation import Activation
from pydtnn.backends.pycuda.abstract.layerable import LayerablePycuda
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn.utils.constants import ArrayShape

__all__ = (
    "ActivationPycuda",
)

logger = logging.getLogger(__name__)


class ActivationPycuda(Activation[TensorArray], LayerablePycuda):
    """
    Extends an Activation class with the attributes and methods required by GPU Activations.

    The next methods are copied from LayerPycuda:
      * reduce_weights_async()
      * wait_allreduce_async()
      * reduce_weights_sync()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The following attributes will be initalized later.
        self.x: TensorArray = None  # type: ignore
        self.dx: TensorArray = None  # type: ignore
        self.grid: tuple[int, int, int] = None  # type: ignore
        self.block: tuple[int, int, int] = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray):
        super()._model_init(prev_shape, x)
        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block
