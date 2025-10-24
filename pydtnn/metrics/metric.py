from pydtnn.backends import PromoteToBackend
from pydtnn.utils.types import ArrayArray
from pydtnn.utils.types import ArrayShape


class Metric[T: ArrayArray](PromoteToBackend):

    def __init__(self, shape: ArrayShape, eps=1e-8):
        self.shape = shape
        self.eps = eps

    def compute(self, y_pred: T, y_targ: T) -> float:
        pass
