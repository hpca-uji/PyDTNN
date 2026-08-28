"""InceptionV3 model architecture implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.initializers import he_uniform

__all__ = ("inceptionv3",)


def inceptionv3(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs an InceptionV3 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers representing the InceptionV3 model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Identity(shape=input_shape))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), stride=2, weights_initializer=he_uniform))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), weights_initializer=he_uniform))
    _(Conv2D(nfilters=64, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))
    _(Conv2D(nfilters=80, filter_shape=(1, 1), weights_initializer=he_uniform))
    _(Conv2D(nfilters=192, filter_shape=(3, 3), weights_initializer=he_uniform))
    _(MaxPool2D(pool_shape=(3, 3), stride=2))

    inception_blocks = [
        [64, 48, 64, 64, 96, 32],
        [64, 48, 64, 64, 96, 64],
        [64, 48, 64, 64, 96, 64],
    ]

    for n1x1, n5x5red, n5x5, n3x3red, n3x3, pool_planes in inception_blocks:
        _(
            ConcatenationBlock(
                [Conv2D(nfilters=n1x1, filter_shape=(1, 1), weights_initializer=he_uniform)],
                [
                    Conv2D(nfilters=n5x5red, filter_shape=(1, 1), weights_initializer=he_uniform),
                    Conv2D(
                        nfilters=n5x5,
                        filter_shape=(5, 5),
                        padding=2,
                        weights_initializer=he_uniform,
                    ),
                ],
                [
                    Conv2D(nfilters=n3x3red, filter_shape=(1, 1), weights_initializer=he_uniform),
                    Conv2D(
                        nfilters=n3x3,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n3x3,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                ],
                [
                    AveragePool2D(pool_shape=(3, 3), stride=1, padding=1),
                    Conv2D(
                        nfilters=pool_planes, filter_shape=(1, 1), weights_initializer=he_uniform
                    ),
                ],
            )
        )

    inception_blocks = [[384, 64, 96]]

    for n1x1, n3x3red, n3x3 in inception_blocks:
        _(
            ConcatenationBlock(
                [
                    Conv2D(
                        nfilters=n1x1,
                        filter_shape=(3, 3),
                        stride=2,
                        padding=0,
                        weights_initializer=he_uniform,
                    )
                ],
                [
                    Conv2D(nfilters=n3x3red, filter_shape=(1, 1), weights_initializer=he_uniform),
                    Conv2D(nfilters=n3x3, filter_shape=(3, 3), weights_initializer=he_uniform),
                    Conv2D(
                        nfilters=n3x3,
                        filter_shape=(3, 3),
                        stride=2,
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                ],
                [MaxPool2D(pool_shape=(3, 3), stride=2, padding=0)],
            )
        )

    inception_blocks = [[192, 128], [192, 160], [192, 160], [192, 192]]

    for n1x1, n1x7 in inception_blocks:
        _(
            ConcatenationBlock(
                [Conv2D(nfilters=n1x1, filter_shape=(1, 1), weights_initializer=he_uniform)],
                [
                    Conv2D(nfilters=n1x7, filter_shape=(1, 1), weights_initializer=he_uniform),
                    Conv2D(
                        nfilters=n1x7,
                        filter_shape=(1, 7),
                        padding=(3, 0),
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n1x1,
                        filter_shape=(7, 1),
                        padding=(0, 3),
                        weights_initializer=he_uniform,
                    ),
                ],
                [
                    Conv2D(nfilters=n1x7, filter_shape=(1, 1), weights_initializer=he_uniform),
                    Conv2D(
                        nfilters=n1x7,
                        filter_shape=(7, 1),
                        padding=(0, 3),
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n1x7,
                        filter_shape=(1, 7),
                        padding=(3, 0),
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n1x7,
                        filter_shape=(7, 1),
                        padding=(0, 3),
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n1x1,
                        filter_shape=(1, 7),
                        padding=(3, 0),
                        weights_initializer=he_uniform,
                    ),
                ],
                [
                    AveragePool2D(pool_shape=(3, 3), stride=1, padding=1),
                    Conv2D(nfilters=n1x1, filter_shape=(1, 1), weights_initializer=he_uniform),
                ],
            )
        )

    inception_blocks = [[192, 320]]

    for n1x1, n3x3 in inception_blocks:
        _(
            ConcatenationBlock(
                [
                    Conv2D(nfilters=n1x1, filter_shape=(1, 1), weights_initializer=he_uniform),
                    Conv2D(
                        nfilters=n3x3, filter_shape=(3, 3), stride=2, weights_initializer=he_uniform
                    ),
                ],
                [
                    Conv2D(
                        nfilters=n1x1,
                        filter_shape=(1, 1),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n1x1,
                        filter_shape=(1, 7),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n1x1,
                        filter_shape=(7, 1),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n1x1, filter_shape=(3, 3), stride=2, weights_initializer=he_uniform
                    ),
                ],
                [MaxPool2D(pool_shape=(3, 3), stride=2)],
            )
        )

    inception_blocks = [[320, 384, 448, 192], [320, 384, 448, 192]]

    for n1x1b0, n1x1b1, n1x1b2, n1x1b3 in inception_blocks:
        _(
            ConcatenationBlock(
                [Conv2D(nfilters=n1x1b0, filter_shape=(1, 1), weights_initializer=he_uniform)],
                [
                    Conv2D(nfilters=n1x1b1, filter_shape=(1, 1), weights_initializer=he_uniform),
                    ConcatenationBlock(
                        [
                            Conv2D(
                                nfilters=n1x1b1,
                                filter_shape=(1, 3),
                                padding=(0, 1),
                                weights_initializer=he_uniform,
                            )
                        ],
                        [
                            Conv2D(
                                nfilters=n1x1b1,
                                filter_shape=(3, 1),
                                padding=(1, 0),
                                weights_initializer=he_uniform,
                            )
                        ],
                    ),
                ],
                [
                    Conv2D(nfilters=n1x1b2, filter_shape=(1, 1), weights_initializer=he_uniform),
                    Conv2D(
                        nfilters=n1x1b1,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    ConcatenationBlock(
                        [
                            Conv2D(
                                nfilters=n1x1b1,
                                filter_shape=(1, 3),
                                padding=(0, 1),
                                weights_initializer=he_uniform,
                            )
                        ],
                        [
                            Conv2D(
                                nfilters=n1x1b1,
                                filter_shape=(3, 1),
                                padding=(1, 0),
                                weights_initializer=he_uniform,
                            )
                        ],
                    ),
                ],
                [
                    AveragePool2D(pool_shape=(3, 3)),
                    Conv2D(
                        nfilters=n1x1b3,
                        filter_shape=(1, 1),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                ],
            )
        )

    _(AveragePool2D(pool_shape=(8, 8), stride=1))  # Global average pooling 2D
    _(Flatten())
    _(FC(shape=(1024,)))
    _(BatchNormalization())
    _(Relu())
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model
