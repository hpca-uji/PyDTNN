import logging
logger = logging.getLogger(__name__)

from pydtnn.backends import PromoteToBackend
from pydtnn.layer_base import LayerBase
from pydtnn.utils import find_component
from pydtnn.utils.constants import Array


class Optimizer[T: Array](PromoteToBackend):
    """
    Optimizer abstract base class
    """

    def __init__(self, learning_rate: float = 1e-2):
        super().__init__()
        self.learning_rate: float = learning_rate
        self.context = dict[int, dict[str, int | T]]()

    def _model_init(self, list_layers: list[LayerBase[T]]) -> None:
        super()._model_init()
        self.dtype = self.model.dtype
        self.gpudirect = self.model.gpudirect

    def update(self, layer: LayerBase) -> None:
        raise NotImplementedError("method update of an Optimizer's child class is not implemented")


def select(name: str) -> type[Optimizer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
