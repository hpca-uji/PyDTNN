from pydtnn.utils.constants import Array, ArrayShape
from pydtnn.layers.layer import Layer
import logging
logger = logging.getLogger(__name__)


class Scalar[T: Array](Layer[T]):
    def __init__(self, shape: ArrayShape = (1,), scale: float = 1.0):
        super().__init__(shape)
        self.scale = scale
