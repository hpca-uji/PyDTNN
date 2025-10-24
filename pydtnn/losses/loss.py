from pydtnn.backends import PromoteToBackend
from pydtnn.utils.types import Array
from pydtnn.utils.types import ArrayShape


class Loss[T: Array](PromoteToBackend):

    def __init__(self, shape: ArrayShape, eps=1e-8):
        self.shape = shape
        self.eps = eps

    def compute(self, y_pred: T, y_targ: T, batch_size: int) -> tuple[float, T]:
        pass
