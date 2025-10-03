from collections.abc import Sequence

from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase

from ..layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from ..activations import relu, softmax


def create_vgg11(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    conv_pattern = [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=relu))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=softmax))

    return model
