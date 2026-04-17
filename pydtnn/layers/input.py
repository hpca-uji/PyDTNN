from pydtnn.utils.constants import Array
from pydtnn.layers.layer import Layer
import logging
logger = logging.getLogger(__name__)


class Input[T: Array](Layer[T]):

    def __init__(self, shape: tuple = (1,)):
        super().__init__(shape)
