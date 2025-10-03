from collections.abc import Sequence, Iterable

from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase

from ..layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from ..activations import relu, softmax
from pydtnn.initializers import he_uniform


def create_vgg19(input_shape: Sequence[int], output_shape: Sequence[int]) -> Iterable[LayerAndActivationBase]:
    yield (Input(shape=input_shape))
    conv_pattern = [[2, 64], [2, 128], [4, 256], [4, 512], [4, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            yield Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=relu, weights_initializer=he_uniform)
        yield MaxPool2D(pool_shape=(2, 2), stride=2)
    yield Flatten()
    yield FC(shape=(4096,), activation=relu, weights_initializer=he_uniform)
    yield Dropout(rate=0.5)
    yield FC(shape=(4096,), activation=relu, weights_initializer=he_uniform)
    yield Dropout(rate=0.5)
    yield FC(shape=output_shape, activation=softmax, weights_initializer=he_uniform)
