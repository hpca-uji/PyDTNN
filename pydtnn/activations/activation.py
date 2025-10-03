from abc import ABC

from ..backends import PromoteToBackendMixin
from ..layers.layer_and_activation_base import LayerAndActivationBase

from numpy import ndarray
from ..backends.gpu.tensor_gpu import TensorGPU

class Activation(PromoteToBackendMixin, LayerAndActivationBase, ABC):

    def __init__(self, shape: tuple[int,...] = (1,)):
        super().__init__(shape)
        self.y: ndarray | TensorGPU = None

    def initialize(self, prev_shape: tuple[int,...]):
        super().initialize(prev_shape)
        self.shape = prev_shape

    @property
    def canonical_name_with_id(self) -> str:
        return f"{self._id_prefix}{self.canonical_name}"
