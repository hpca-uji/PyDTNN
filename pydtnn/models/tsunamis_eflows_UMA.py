from collections.abc import Sequence

from pydtnn.activations.relu import Relu
from pydtnn.activations.sigmoid import Sigmoid
from pydtnn.layer_base import LayerBase
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.initializers import he_uniform


def tsunamis_eflows_UMA(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform))

    UMA_blocks = [[32, 64, 64],
                  [64, 128, 128],
                  [128, 256, 256]]

    for n3x3, n3x3red, n2x2 in UMA_blocks:
        _(ConcatenationBlock(
            [Conv2D(nfilters=n3x3, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform),
             Conv2D(nfilters=n3x3red, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform),
             MaxPool2D(pool_shape=(3, 3), stride=2, padding=1)
             ],
            [Conv2D(nfilters=n2x2, filter_shape=(3, 3), padding=1, stride=2, weights_initializer=he_uniform)
             ]))

    UMA_dense_blocks = [[128, 256, 512],
                        [128, 256, 512]]

    for n3x3, n3x3red, n3x3fin in UMA_dense_blocks:
        _(ConcatenationBlock(
            [Conv2D(nfilters=n3x3, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform),
             Conv2D(nfilters=n3x3red, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform),
             Conv2D(nfilters=n3x3fin, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform)
             ], []))

    _(AveragePool2D(pool_shape=(10, 10), stride=1))  # Global average pooling 2D
    _(Flatten())

    for fc in range(3):
        _(FC(shape=(500,), activation=Relu))

    _(FC(shape=output_shape, activation=Sigmoid))

    return model
