from collections.abc import Sequence

from pydtnn.activations import *
from pydtnn.layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.initializers import he_uniform
from pydtnn.activations import softmax


def create_resnet1202(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=16, filter_shape=(3, 3), stride=1, padding=1, weights_initializer=he_uniform))
    _(BatchNormalization())

    layout = [[16, 200, 1], [32, 200, 2], [64, 200, 2]]  # Resnet-1202
    for n_filt, res_blocks, stride in layout:
        for r in range(res_blocks):
            if r > 0:
                stride = 1
            _(AdditionBlock(
                [
                    Conv2D(nfilters=n_filt, filter_shape=(3, 3), stride=stride, padding=1,
                           weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n_filt, filter_shape=(3, 3), stride=1, padding=1,
                           weights_initializer=he_uniform),
                    BatchNormalization()
                ],
                [
                    Conv2D(nfilters=n_filt, filter_shape=(1, 1), stride=stride, weights_initializer=he_uniform),
                    BatchNormalization()
                ] if r == 0 or stride != 1 else []))
            _(Relu())

    _(AveragePool2D(pool_shape=(0, 0)))  # Global average pooling 2D
    _(Flatten())
    _(FC(shape=(64,)))
    _(BatchNormalization())
    _(Relu())
    _(FC(shape=output_shape, activation=softmax))

    return model
