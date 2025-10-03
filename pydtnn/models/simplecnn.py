from collections.abc import Sequence

from ..layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from ..activations import relu, softmax

def create_simplecnn(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=4, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
    _(Conv2D(nfilters=8, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation=relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=softmax))

    return model
