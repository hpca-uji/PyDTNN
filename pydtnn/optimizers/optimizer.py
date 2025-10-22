from abc import ABC, abstractmethod

from pydtnn.backends import PromoteToBackendMixin
import numpy as np

from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase


class Optimizer(PromoteToBackendMixin, ABC):
    """
    Optimizer abstract base class
    """

    def __init__(self, learning_rate: float = 1e-2, dtype: np.dtype = np.float32):
        super().__init__()
        self.learning_rate: float = learning_rate
        self.dtype: np.dtype = dtype
        self.context: dict = dict()
        # Only for GPU implementations:
        self.real_batch_size: int = None

    @abstractmethod
    def initialize(self, list_layers: list[LayerAndActivationBase]) -> None:
        raise NotImplementedError("method \"initialize\" of an Optimizer's child class is not implemented")

    @abstractmethod
    def update(self, layer: LayerAndActivationBase) -> None:
        pass
