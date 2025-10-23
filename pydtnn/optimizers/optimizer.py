from pydtnn.backends import PromoteToBackend
import numpy as np

from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase


class Optimizer(PromoteToBackend):
    """
    Optimizer abstract base class
    """

    def __init__(self, learning_rate: float = 1e-2, dtype: np.dtype = np.float32):
        super().__init__()
        self.learning_rate: float = learning_rate
        self.dtype: np.dtype = dtype
        self.context: dict = dict()

    def initialize(self, list_layers: list[LayerAndActivationBase]) -> None:
        raise NotImplementedError("method \"initialize\" of an Optimizer's child class is not implemented")

    def update(self, layer: LayerAndActivationBase) -> None:
        pass
