"""Numpy implementation of the Encoder-Decoder architecture."""

import logging
from typing import TYPE_CHECKING, Any

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.backends.numpy.layers.decoder import Decoder
from pydtnn.backends.numpy.layers.encoder import Encoder
from pydtnn.layers.encoder_decoder import EncoderDecoder
from pydtnn.libs import numpy as np
from pydtnn.utils.constants import ArrayShape

__all__ = ("EncoderDecoderNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class EncoderDecoderNumpy(EncoderDecoder[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy-based Encoder-Decoder block layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the EncoderDecoderNumpy layer with encoder and decoder stacks."""
        super().__init__(*args, **kwargs)
        self.encoder = [
            Encoder[np.ndarray](
                embedl=self.embedl,
                d_k=self.d_k,
                d_ff=self.d_ff,
                heads=self.heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.enc_layers)
        ]
        self.decoder = [
            Decoder[np.ndarray](
                embedl=self.embedl,
                d_k=self.d_k,
                d_ff=self.d_ff,
                heads=self.heads,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.dec_layers)
        ]
        self.paths = [self.encoder + self.decoder]  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initializes model parameters and sublayers based on input shapes."""
        super()._model_init(prev_shape, x)
        if len(prev_shape) == 2:
            x_enc, x_dec = x if x else (None, None)
            mask_enc = mask_dec = None
            enc_shape = (prev_shape[0], ())
            dec_shape = (prev_shape[0], prev_shape[1], ())
        else:
            x_enc, mask_enc, x_dec, mask_dec = x if x else (None, None, None, None)
            enc_shape = (prev_shape[0], prev_shape[1])
            dec_shape = (prev_shape[2], prev_shape[0], prev_shape[3])
        self.embedl = enc_shape[0][-1]  # type: ignore (This layer is special)

        # Initialize all sublayers
        for layer in self.children:
            layer._init_backend_with_model(self.model)

        # type: ignore (encoder has multiple parameters)
        self.encoder[0]._model_init(prev_shape=enc_shape, x=(x_enc, mask_enc))  # type: ignore
        for layer in self.encoder[1:]:
            # type: ignore (encoder has multiple parameters)
            layer._model_init(prev_shape=enc_shape, x=(x_enc, mask_enc))  # type: ignore
        for layer in self.decoder:
            # type: ignore (encoder has multiple parameters)
            layer._model_init(prev_shape=dec_shape, x=(x_dec, x_enc, mask_dec))  # type: ignore

        for layer in self.children:
            self.fwd_time += layer.fwd_time
            self.bwd_time += layer.bwd_time
            self.nparams += layer.nparams

    def initialize_block_layer(self) -> None:
        """Placeholder for block layer initialization logic."""
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the encoder and decoder stacks."""
        if len(x) == 2:
            x, y = x
            x_mask = y_mask = None
        else:
            x, x_mask, y, y_mask = x
        # print(x.shape, x_mask.shape, y.shape, y_mask.shape)
        for i in range(self.enc_layers):  # Encoding layers
            x = self.encoder[i].forward(x, x_mask)  # type: ignore (encoder has multiple parameters)
        for i in range(self.dec_layers):  # Decoding layers
            # type: ignore (encoder has multiple parameters)
            y = self.decoder[i].forward(y, x, y_mask)  # type: ignore (In transformer's layers is fine)
        self.y = y
        return y

    def backward(self, prev_dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs the backward pass through the decoder and encoder stacks."""
        dx_tgt = prev_dx
        dx_enc: np.ndarray = 0.0  # type: ignore (It'll be a np.ndarray)
        for i in range(self.dec_layers):  # Decoding layers
            dx_tgt, dx2 = self.decoder[-1 * (i + 1)].backward(dx_tgt)
            dx_enc += dx2
        for i in range(self.enc_layers):  # Enconding layers
            # type: ignore (encoder has multiple parameters)
            dx_enc = self.encoder[-1 * (i + 1)].backward(dx_enc)
        # if self.need_dx:
        return dx_tgt, dx_enc
