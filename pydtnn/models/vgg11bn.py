from collections.abc import Sequence

from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase

from ..activations import *
from ..layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.initializers import he_uniform


def create_vgg11bn(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    conv_pattern = [[1, 64], [1, 128], [2, 256], [2, 512], [2, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, weights_initializer=he_uniform))
            _(BatchNormalization())
            _(Relu())
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(Dropout(rate=0.5))
    _(FC(shape=(512,), weights_initializer=he_uniform))
    _(BatchNormalization())
    _(Relu())
    _(Dropout(rate=0.5))
    _(FC(shape=(512,), weights_initializer=he_uniform))
    _(BatchNormalization())
    _(Relu())
    _(FC(shape=output_shape, activation=softmax, weights_initializer=he_uniform))

    return model
