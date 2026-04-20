from pydtnn.utils.constants import Array, ArrayShape
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
import logging
logger = logging.getLogger(__name__)


class FeedForward[T: Array](AbstractBlockLayer[T]):
    def __init__(self, shape: ArrayShape = (1,), d_ff: int = 256, dropout_rate: float = 0.5):
        super().__init__()
        self.shape = shape
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
