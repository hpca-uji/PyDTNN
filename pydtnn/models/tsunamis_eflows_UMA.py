"""Module for the Tsunamis Eflows UMA model architecture."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.log_sigmoid import LogSigmoid
from pydtnn.activations.relu import Relu
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.initializers import he_uniform

__all__ = ("tsunamis_eflows_uma",)


def tsunamis_eflows_uma(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs the Tsunamis Eflows UMA model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output data.

    Returns:
        A sequence of layers defining the model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Identity(shape=input_shape))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform))
    _(Conv2D(nfilters=32, filter_shape=(3, 3), padding=1, weights_initializer=he_uniform))

    uma_blocks = [[32, 64, 64], [64, 128, 128], [128, 256, 256]]

    for n3x3, n3x3red, n2x2 in uma_blocks:
        _(
            ConcatenationBlock(
                [
                    Conv2D(
                        nfilters=n3x3,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n3x3red,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    MaxPool2D(pool_shape=(3, 3), stride=2, padding=1),
                ],
                [
                    Conv2D(
                        nfilters=n2x2,
                        filter_shape=(3, 3),
                        padding=1,
                        stride=2,
                        weights_initializer=he_uniform,
                    )
                ],
            )
        )

    uma_dense_blocks = [[128, 256, 512], [128, 256, 512]]

    for n3x3, n3x3red, n3x3fin in uma_dense_blocks:
        _(
            ConcatenationBlock(
                [
                    Conv2D(
                        nfilters=n3x3,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n3x3red,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                    Conv2D(
                        nfilters=n3x3fin,
                        filter_shape=(3, 3),
                        padding=1,
                        weights_initializer=he_uniform,
                    ),
                ],
                [],
            )
        )

    _(AveragePool2D(pool_shape=(10, 10), stride=1))  # Global average pooling 2D
    _(Flatten())

    for fc in range(3):
        _(FC(shape=(500,), activation=Relu))

    _(FC(shape=output_shape, activation=LogSigmoid))

    return model
