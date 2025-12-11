from collections.abc import Sequence

from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase

from pydtnn.layer_base import LayerBase
from pydtnn.utils.initializers import he_uniform
from pydtnn.layers.max_pool_2d import MaxPool2D


def vgg11bn(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
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
    _(FC(shape=output_shape, activation=Softmax, weights_initializer=he_uniform))

    return model
