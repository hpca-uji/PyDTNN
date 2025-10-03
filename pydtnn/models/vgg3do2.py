from collections.abc import Sequence

from pydtnn.layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.initializers import he_uniform
from pydtnn.activations import relu, softmax


def create_vgg3do2(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Dropout(rate=0.2))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Dropout(rate=0.3))
    _(Conv2D(nfilters=128, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=128, filter_shape=(3, 3), padding=1, activation=relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Dropout(rate=0.4))
    _(Flatten())
    _(FC(shape=(128,), activation=relu, weights_initializer=he_uniform))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=softmax))

    return model
