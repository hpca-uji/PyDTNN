from collections.abc import Sequence

from pydtnn.activations.relu import Relu
from pydtnn.layer_base import LayerBase
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.max_pool_2d import MaxPool2D


def vgg_cyclone(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))
    conv_pattern = [[3, 3, 1, 64], [2, 3, 1, 128], [2, 3, 1, 256], [3, 3, 1, 512]]
    for nlayers, filter_, padding_, nfilters_ in conv_pattern:
        for layer in range(nlayers):
            _(Conv2D(nfilters=nfilters_, filter_shape=(filter_, filter_), padding=padding_, stride=1, activation=Relu))
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(FC(shape=(256,), activation=Relu))
    _(FC(shape=(128,), activation=Relu))
    _(FC(shape=(64,), activation=Relu))
    _(FC(shape=output_shape))

    return model
