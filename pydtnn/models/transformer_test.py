"""
Module for testing and defining transformer-based model architectures within the PyDTNN framework.
"""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.layers.input import Input
from pydtnn.layers.multi_head_attention import MultiHeadAttention
from pydtnn.utils.constants import ArrayShape

__all__ = ("transformer_test",)


def transformer_test(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a sequence of layers representing a transformer-based model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The expected shape of the output data.

    Returns:
        A sequence of Layerable objects forming the model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=((1, 75, 300))))
    _(MultiHeadAttention(embedl=300, d_k=8, heads=4, dropout_rate=0.0))
    # _(LayerNormalization())
    # _(EncoderDecoder(enc_layers=6, dec_layers=6, embedl=300, d_k=30, heads=10, d_ff=1200, dropout_rate=0.0))

    return model
