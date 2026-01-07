"""
https://github.com/google-research/bert
https://github.com/google-research/bert/blob/master/modeling.py
https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert
https://huggingface.co/datasets/wikipedia
https://huggingface.co/albert-base-v2

d_k = embeld / heads

Tensorflow parameters names:
    hidden_size -> embedl
    intermediate_size -> d_ff
    num_hidden_layers -> n_encoders
    The rest of the parameters should be obvious
"""

from collections.abc import Sequence
from pydtnn.layer_base import LayerBase
from pydtnn.layers.encoder import Encoder
from pydtnn.layers.input import Input


def bert(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    """Bert-Medium"""
    model = list[LayerBase]()
    _ = model.append

    n_encoders = 8
    max_seq = 512
    embedl = 512

    _(Input(shape=(1, max_seq, embedl)))
    for i in range(n_encoders):
        _(Encoder(embedl=embedl, d_k=64, heads=8, d_ff=2048, dropout_rate=0.1))

    return model

# def create_bert(model):
#     n_encoders = 3
#     _ = model.add
#     _(Input(shape=(1,75,300)))
#     for i in range(n_encoders):
#         _(Encoder(embedl=300, d_k=30, heads=10, d_ff=1200, dropout_rate=0.1))
