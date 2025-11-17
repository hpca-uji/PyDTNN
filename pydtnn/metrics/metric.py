from abc import abstractmethod

from pydtnn.backends import PromoteToBackend
from pydtnn.utils import find_component
from pydtnn.utils.constants import ArrayShape, Array
import numpy as np

class Metric[T: Array](PromoteToBackend):
    format = ""
    order = 0   # No need of special order.

    def __init__(self, shape: ArrayShape, eps=1e-8):
        self.shape = shape
        self.eps = eps

    def initialize(self) -> None:
        pass

    @abstractmethod
    def compute(self, y_pred: T, y_targ: T) -> float | np.ndarray:
        pass


def select(name: str) -> type[Metric]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
