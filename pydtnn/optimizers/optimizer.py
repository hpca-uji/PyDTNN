import logging
logger = logging.getLogger(__name__)

from pydtnn.abstract.base import Base
from pydtnn.abstract.layerable import Layerable
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array


class Optimizer[T: Array](Base):
    """
    Optimizer abstract base class
    """

    def __init__(self, learning_rate: float = 1e-2):
        super().__init__()
        self.learning_rate: float = learning_rate
        self.context = dict[int, dict[str, int | T]]()

    def _model_init(self, list_layers: list[Layerable[T]]) -> None:
        super()._model_init()
        self.dtype = self.model.dtype
        self.gpudirect = self.model.gpudirect

    def update(self, layer: Layerable) -> None:
        raise NotImplementedError("method update of an Optimizer's child class is not implemented")


def select(name: str) -> type[Optimizer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
