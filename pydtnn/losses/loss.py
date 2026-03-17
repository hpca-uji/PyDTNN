import logging
logger = logging.getLogger(__name__)

from pydtnn.backends import PromoteToBackend
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array


class Loss[T: Array](PromoteToBackend):
    format = ""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _model_init(self) -> None:
        super()._model_init()
        self.shape = self.model._output_shape

    def compute(self, y_pred: T, y_targ: T, batch_size: int) -> tuple[float, T]:
        raise NotImplementedError()


def select(name: str) -> type[Loss]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
