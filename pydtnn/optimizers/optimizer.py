import numpy as np

from pydtnn.backends import PromoteToBackend
from pydtnn.layer_base import LayerBase
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array


class Optimizer[T: Array](PromoteToBackend):
    """
    Optimizer abstract base class
    """

    def __init__(self, learning_rate: float = 1e-2, dtype: np.dtype = np.dtype(np.float32)):
        super().__init__()
        self.learning_rate: float = learning_rate
        self.dtype: np.dtype = dtype
        self.context = dict[int, dict[str, int | T]]()

    def initialize(self, list_layers: list[LayerBase]) -> None:
        raise NotImplementedError("method \"initialize\" of an Optimizer's child class is not implemented")

    def update(self, layer: LayerBase) -> None:
        raise NotImplementedError("method \"update\" of an Optimizer's child class is not implemented")


def select(name: str) -> type[Optimizer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
