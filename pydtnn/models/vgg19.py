from collections.abc import Sequence, Iterable

from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase

from pydtnn.utils.initializers import he_uniform
from pydtnn.layers.max_pool_2d import MaxPool2D


def vgg19(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))
    conv_pattern = [[2, 64], [2, 128], [4, 256], [4, 512], [4, 512]]
    for nlayers, nfilters in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters, filter_shape=(3, 3), padding=1, stride=1, activation=Relu, weights_initializer=he_uniform))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=Relu, weights_initializer=he_uniform))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=Relu, weights_initializer=he_uniform))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=Softmax, weights_initializer=he_uniform))

    return model
