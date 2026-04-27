import logging

from pydtnn.activations.activation import Activation
from pydtnn.utils.constants import Array, ArrayShape

logger = logging.getLogger(__name__)


class Relu[T: Array](Activation[T]):

    def __init__(self, shape: ArrayShape = (1,)):
        super().__init__(shape)
        # Will be initalized in "initialize"
        self.mask: T = None  # type: ignore
