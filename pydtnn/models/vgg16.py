from collections.abc import Sequence

from ..layers import *
from ..activations import relu, softmax
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase


def create_vgg16(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))

    conv_pattern = [[2, 64], [2, 128], [3, 256], [3, 512], [3, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))

    _(Flatten())
    _(FC(shape=(4096,), activation=relu))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=relu))
    _(Dropout(rate=0.5))
    _(FC(output_shape, activation=softmax))

    return model
