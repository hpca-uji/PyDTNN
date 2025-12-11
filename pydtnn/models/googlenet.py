from collections.abc import Sequence

from pydtnn.activations.softmax import Softmax
from pydtnn.activations.relu import Relu
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase
from pydtnn.utils.initializers import he_uniform
from pydtnn.layers.max_pool_2d import MaxPool2D


def googlenet(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    model = list[LayerBase]()
    _ = model.append

    _(Input(shape=input_shape))

    _(Conv2D(nfilters=192, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform))
    _(BatchNormalization())
    _(Relu())

    inception_blocks = [[64, 96, 128, 16, 32, 32],
                        [128, 128, 192, 32, 96, 64],
                        [],
                        [192, 96, 208, 16, 48, 64],
                        [160, 112, 224, 24, 64, 64],
                        [128, 128, 256, 24, 64, 64],
                        [112, 144, 288, 32, 64, 64],
                        [256, 160, 320, 32, 128, 128],
                        [],
                        [256, 160, 320, 32, 128, 128],
                        [384, 192, 384, 48, 128, 128]]

    for layout in inception_blocks:
        if not layout:
            _(MaxPool2D(pool_shape=(3, 3), stride=2, padding=1))
        else:
            n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes = layout
            _(ConcatenationBlock(
                [
                    # 1x1 conv branch
                    Conv2D(nfilters=n1x1, filter_shape=(1, 1), weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu()
                ],
                [  # 1x1 conv -> 3x3 conv branch
                    Conv2D(nfilters=n3x3red, filter_shape=(1, 1), weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n3x3, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu()
                ],
                [  # 1x1 conv -> 5x5 conv branch
                    Conv2D(nfilters=n5x5red, filter_shape=(1, 1), weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n5x5, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n5x5, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu()
                ],
                [  # 3x3 pool -> 1x1 conv branch
                    MaxPool2D(pool_shape=(3, 3), stride=1, padding=1),
                    Conv2D(nfilters=pool_planes, filter_shape=(1, 1), weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu()
                ]))

    _(AveragePool2D(pool_shape=(8, 8), stride=1))  # Global average pooling 2D
    _(Flatten())
    _(FC(shape=(1024,)))
    _(BatchNormalization())
    _(Relu())
    _(FC(shape=output_shape, activation=Softmax))

    return model
