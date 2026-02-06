from abc import abstractmethod

from pydtnn.backends import PromoteToBackend
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array
import numpy as np


class Metric[T: Array](PromoteToBackend):
    format = ""
    order = 0   # No need of special order.

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def initialize(self) -> None:
        self.shape = self.model._output_shape

    def post_initialize(self) -> None:
        """
        Method were the operations that requiere a initialize are done.
        """
        pass

    @abstractmethod
    def compute(self, y_pred: T, y_targ: T) -> float | np.ndarray:
        pass


def select(name: str) -> type[Metric]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
