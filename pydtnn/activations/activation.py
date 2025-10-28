from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.utils.types import ArrayShape, Array


class Activation[T: Array](LayerAndActivationBase):

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)
        self.y: T = None  #type: ignore (it will be initalized later)

    def initialize(self, prev_shape: ArrayShape, x: T | None = None):
        super().initialize(prev_shape, x)
        self.shape = prev_shape

    @property
    def canonical_name_with_id(self) -> str:
        return f"{self._id_prefix}{self.canonical_name}"
