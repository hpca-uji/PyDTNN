from abc import ABC

from pydtnn.backends import PromoteToBackendMixin
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.utils.types import Array

class Activation[T: Array](PromoteToBackendMixin, LayerAndActivationBase, ABC):

    def __init__(self, shape: tuple[int, ...] = (1,)):
        super().__init__(shape)
        self.y: T = None

    def initialize(self, prev_shape: tuple[int, ...], x: T | None = None):
        super().initialize(prev_shape, x)
        self.shape = prev_shape

    @property
    def canonical_name_with_id(self) -> str:
        return f"{self._id_prefix}{self.canonical_name}"
