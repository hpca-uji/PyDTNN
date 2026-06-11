"""ResNet-18 model architecture implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.initializers import he_uniform

__all__ = ("resnet10",)


def resnet10(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a ResNet-10 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers representing the ResNet-10 model.
    """
    model = list[Layerable]()
    _ = model.append

    first_filters = 32
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), stride=1, padding=1, weights_initializer=he_uniform))
    _(BatchNormalization())

    layout = [[first_filters, 1, 1], [first_filters * 2, 1, 2], [first_filters * 4 , 1, 2], [first_filters * 8, 1, 2]]  # Resnet-10
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
    _(FC(shape=(512,)))
    _(BatchNormalization())
    _(Relu())
    _(FC(shape=output_shape, activation=Softmax))

    return model
