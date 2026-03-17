import logging
logger = logging.getLogger(__name__)

from pydtnn.layer_base import LayerBase
from pydtnn.utils import find_component
from pydtnn.utils.constants import ArrayShape, Array


class Activation[T: Array](LayerBase):

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)

    def _model_init(self, prev_shape: ArrayShape, x: T | None = None):
        super()._model_init(prev_shape, x)
        self.shape = prev_shape


def select(name: str) -> type[Activation]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
