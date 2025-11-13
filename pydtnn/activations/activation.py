from pydtnn.layer import LayerAndActivationBase
from pydtnn.utils import find_component
from pydtnn.utils.types import ArrayShape, Array

from typing import Self

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


def select(name: str) -> type[Activation]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
