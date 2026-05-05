# https://aclanthology.org/W18-2509.pdf


from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.encoder_decoder import EncoderDecoder
from pydtnn.layers.input import Input
from pydtnn.utils.constants import ArrayShape

__all__ = ("iwslt",)


def iwslt(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=((1, 512, 512), (1, 512, 512))))
    _(EncoderDecoder(enc_layers=6, dec_layers=6, embedl=512, d_k=64, heads=8, d_ff=2048, dropout_rate=0.1))

    return model
