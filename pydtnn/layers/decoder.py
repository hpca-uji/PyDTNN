from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
from pydtnn.utils.constants import Array


class Decoder[T: Array](AbstractBlockLayer[T]):
    def __init__(self, embedl: int = 64, d_k: int = 3, d_ff: int = 256, heads: int = 10, dropout_rate: float = 0.5):
        super().__init__()
        self.embedl = embedl
        self.heads = heads
        self.d_k = d_k
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
