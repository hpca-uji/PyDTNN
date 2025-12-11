from collections.abc import Sequence

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.max_pool_2d import MaxPool2D


def alexnet(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=96, filter_shape=(11, 11), padding=0, stride=4, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=256, filter_shape=(5, 5), padding=2, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(Conv2D(nfilters=384, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(Conv2D(nfilters=256, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Flatten())
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=(4096,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=Softmax))

    return model
