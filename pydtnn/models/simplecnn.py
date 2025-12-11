from collections.abc import Sequence

from pydtnn.layers.input import Input
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.fc import FC
from pydtnn.layers.dropout import Dropout
from pydtnn.layer_base import LayerBase
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax


def simplecnn(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=4, filter_shape=(3, 3), padding=1, stride=1, activation=Relu, grouping=Conv2D.Grouping.STANDARD))
    _(Conv2D(nfilters=8, filter_shape=(3, 3), padding=1, stride=1, activation=Relu))
    _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(128,), activation=Relu))
    _(Dropout(rate=0.5))
    _(FC(shape=output_shape, activation=Softmax))

    return model
