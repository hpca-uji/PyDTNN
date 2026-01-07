from collections.abc import Sequence
from pydtnn.layer_base import LayerBase
from pydtnn.layers.input import Input
from pydtnn.layers.multi_head_attention import MultiHeadAttention


def transformer_test(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=((1, 75, 300))))
    _(MultiHeadAttention(embedl=300, d_k=8, heads=4, dropout_rate=0.0))
    # _(LayerNormalization())
    # _(EncoderDecoder(enc_layers=6, dec_layers=6, embedl=300, d_k=30, heads=10, d_ff=1200, dropout_rate=0.0))

    return model
