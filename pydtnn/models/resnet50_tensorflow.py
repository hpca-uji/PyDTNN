"""
ResNet50 architecture implementation compatible with TensorFlow specifications.
"""

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
from pydtnn.layers.max_pool_2d import MaxPool2D
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.initializers import he_uniform

# NOTE: PyDTNN follows PyTorch's definitions
# NOTE: TensorFlow uses BatchNormalization with 1.001e-5 epsilon and 0.99 momentum
# NOTE: TensorFlow uses AveragePool2D with (2, 2) pool shape
# NOTE: TensorFlow uses FC with 1024 shape


__all__ = ("resNet50",)


def resNet50(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a ResNet50 model architecture.

    Args:
        input_shape: The shape of the input tensor.
        output_shape: The shape of the output tensor.

    Returns:
        A sequence of layers representing the ResNet50 model.
    """
    model = list[Layerable]()
    _ = model.append
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=64, filter_shape=(7, 7), stride=2, padding=3, weights_initializer=he_uniform))
    _(BatchNormalization())
    _(Relu())
    _(MaxPool2D(pool_shape=(3, 3), stride=2, padding=1))

    expansion = 4
    layout = [[64, 3, 1], [128, 4, 2], [256, 6, 2], [512, 3, 2]]
    for n_filt, res_blocks, stride in layout:
        for r in range(res_blocks):
            if r > 0:
                stride = 1
            _(
                AdditionBlock(
                    [
                        Conv2D(
                            nfilters=n_filt,
                            filter_shape=(1, 1),
                            stride=1,
                            weights_initializer=he_uniform,
                        ),
                        BatchNormalization(),
                        Relu(),
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
                            nfilters=n_filt * expansion,
                            filter_shape=(1, 1),
                            stride=1,
                            weights_initializer=he_uniform,
                        ),
                        BatchNormalization(),
                    ],
                    (
                        [
                            Conv2D(
                                nfilters=n_filt * expansion,
                                filter_shape=(1, 1),
                                stride=stride,
                                weights_initializer=he_uniform,
                            ),
                            BatchNormalization(),
                        ]
                        if r == 0 or stride != 1
                        else []
                    ),
                )
            )
            _(Relu())

    _(AveragePool2D(pool_shape=(1, 1)))
    _(Flatten())
    _(FC(shape=(512 * expansion,), activation=Relu))
    _(FC(shape=output_shape, activation=Softmax))

    return model


resnet = resNet50
