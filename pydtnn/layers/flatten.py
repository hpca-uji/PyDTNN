import logging
import math

from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import Array, ArrayShape

__all__ = (
    "Flatten",
)

logger = logging.getLogger(__name__)


class Flatten[T: Array](Layer[T]):
    def _model_init(self, prev_shape: ArrayShape, x: T | None):
        super()._model_init(prev_shape, x)
        self.shape = (int(math.prod(prev_shape)),)
