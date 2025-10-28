import abc
import numpy as np

from pydtnn.backends import PromoteToBackend
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.utils.types import Array


class Optimizer[T: Array](PromoteToBackend):
    """
    Optimizer abstract base class
    """

    def __init__(self, learning_rate: float = 1e-2, dtype: np.dtype = np.dtype(np.float32)):
        super().__init__()
        self.learning_rate: float = learning_rate
        self.dtype: np.dtype = dtype        
        self.context = dict[int, dict[str, int | T]]()

    @abc.abstractmethod
    def initialize(self, list_layers: list[LayerAndActivationBase]) -> None:
        raise NotImplementedError("method \"initialize\" of an Optimizer's child class is not implemented")

    @abc.abstractmethod
    def update(self, layer: LayerAndActivationBase) -> None:
        raise NotImplementedError("method \"update\" of an Optimizer's child class is not implemented")
