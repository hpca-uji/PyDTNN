from pydtnn.backends import PromoteToBackend
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array
from pydtnn.utils.constants import ArrayShape


class Loss[T: Array](PromoteToBackend):
    format = ""

    def __init__(self, shape: ArrayShape, eps=1e-8):
        self.shape = shape
        self.eps = eps
        self.real_memory_size: int = 0
        self.temp_memory_size: int = 0

    def initialize(self) -> None:
        pass

    def compute(self, y_pred: T, y_targ: T, batch_size: int) -> tuple[float, T]:
        raise NotImplementedError()

    def post_initialize(self) -> None:
        """
        Method were the operations that requiere a initialize are done.
        """
        pass


def select(name: str) -> type[Loss]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
