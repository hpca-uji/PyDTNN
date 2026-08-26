"""ResNet-44 model architecture implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.initializers import he_uniform

__all__ = ("resnet44",)


def resnet44(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a ResNet-44 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers defining the ResNet-44 model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=input_shape))
    _(Conv2D(nfilters=16, filter_shape=(3, 3), stride=1, padding=1, weights_initializer=he_uniform))
    _(BatchNormalization())

    layout = [[16, 9, 1], [32, 9, 2], [64, 9, 2]]  # Resnet-44
    for n_filt, res_blocks, stride in layout:
        for r in range(res_blocks):
            if r > 0:
                stride = 1
            _(
                AdditionBlock(
                    [
                        Conv2D(
                            nfilters=n_filt,
                            filter_shape=(3, 3),
                            stride=stride,
                            padding=1,
                            weights_initializer=he_uniform,
                        ),
                        BatchNormalization(),
                        Relu(),
                        Conv2D(
                            nfilters=n_filt,
                            filter_shape=(3, 3),
                            stride=1,
                            padding=1,
                            weights_initializer=he_uniform,
                        ),
                        BatchNormalization(),
                    ],
                    (
                        [
                            Conv2D(
                                nfilters=n_filt,
                                filter_shape=(1, 1),
                                stride=stride,
                                weights_initializer=he_uniform,
                            ),
                            BatchNormalization(),
                        ]
                        if stride != 1
                        else []
                    ),
                )
            )
            _(Relu())

    _(AveragePool2D(pool_shape=(0, 0)))  # Global average pooling 2D
    _(Flatten())
    _(FC(shape=(64,)))
    _(BatchNormalization())
    _(Relu())
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model
