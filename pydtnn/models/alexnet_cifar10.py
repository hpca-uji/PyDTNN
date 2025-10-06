from collections.abc import Sequence

from pydtnn.layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.activations import relu, softmax


def create_alexnet_cifar10(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, stride=2, activation=relu))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Conv2D(nfilters=192, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
    _(Conv2D(nfilters=256, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
    _(Conv2D(nfilters=256, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=relu))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=softmax))

    return model
