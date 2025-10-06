from collections.abc import Sequence

from pydtnn.layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.activations import relu, softmax


def create_simplemlp(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Flatten())
    _(FC(shape=(512,), activation=relu))
    _(FC(shape=(512,), activation=relu))
    _(FC(shape=(512,), activation=relu))
    _(FC(shape=output_shape, activation=softmax))

    return model
