from collections.abc import Sequence

from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase
from pydtnn.utils.initializers import he_uniform
from pydtnn.layers.max_pool_2d import MaxPool2D


def vgg3(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Conv2D(nfilters=128, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(Conv2D(nfilters=128, filter_shape=(3, 3), padding=1, activation=Relu, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation=Relu, weights_initializer=he_uniform))
    _(FC(shape=output_shape, activation=Softmax))

    return model
