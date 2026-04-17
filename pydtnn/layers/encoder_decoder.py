from pydtnn.utils.constants import Array
from pydtnn.layers.abstract.block_layer import AbstractBlockLayer
import logging
logger = logging.getLogger(__name__)


class EncoderDecoder[T: Array](AbstractBlockLayer[T]):
    def __init__(self, enc_layers: int = 1, dec_layers: int = 1, embedl: int = 64, d_k: int = 3, heads: int = 10, d_ff: int = 256, dropout_rate: float = 0.5):
        super().__init__()
        self.embedl = embedl
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.heads = heads
        self.d_k = d_k
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.encoder = [None,]
        self.decoder = [None,]
        self.paths = [self.encoder + self.decoder]

    def _model_init(self, prev_shape, x):
        super()._model_init(prev_shape, x)

        if self.shape == ():
            self.shape = prev_shape[0]

    def _show_props(self) -> dict:
        props = super()._show_props()

        props["encodes"] = self.enc_layers
        props["decodes"] = self.dec_layers

        return props
