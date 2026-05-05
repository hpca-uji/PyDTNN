from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.input import Input
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.utils.constants import ArrayShape

__all__ = (
    "transformer_test",
)


def transformer_test(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=((1, 75, 300))))
    _(MultiHeadAttention(embedl=300, d_k=8, heads=4, dropout_rate=0.0))
    # _(LayerNormalization())
    # _(EncoderDecoder(enc_layers=6, dec_layers=6, embedl=300, d_k=30, heads=10, d_ff=1200, dropout_rate=0.0))

    return model
