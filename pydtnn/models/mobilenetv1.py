"""MobileNetV1 architecture implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.activations.relu import Relu
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.identity import Identity
from pydtnn.utils.constants import ArrayShape

# NOTE: PyDTNN follows PyTorch's definitions
# NOTE: TensorFlow uses BatchNormalization with 1.001e-5 epsilon and 0.99 momentum
# NOTE: TensorFlow uses AveragePool2D with (2, 2) pool shape
# NOTE: TensorFlow uses FC with 1024 shape
# NOTE: TensorFlow uses LeakyReLU


__all__ = ("mobilenetv1",)


def mobilenetv1(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a MobileNetV1 model architecture.

    Args:
        input_shape: The shape of the input tensor.
        output_shape: The shape of the output tensor.

    Returns:
        A sequence of layers representing the MobileNetV1 model.
    """
    first_filters = 32

    model = list[Layerable]()
    _ = model.append
    _(Identity(shape=input_shape))
    _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), padding=1, stride=2, use_bias=False))
    _(BatchNormalization())
    _(Relu())

    layout = [[64, 1], [128, 2], [256, 2], [512, 6], [1024, 2]]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            _(
                Conv2DDepthwise(
                    nfilters=first_filters,
                    filter_shape=(3, 3),
                    padding=1,
                    stride=stride,
                    use_bias=False,
                )
            )
            _(BatchNormalization())
            _(Relu())
            _(Conv2DPointwise(nfilters=n_filt, filter_shape=(1, 1), use_bias=False))
            _(BatchNormalization())
            _(Relu())
            first_filters = n_filt

    _(AveragePool2D(pool_shape=(1, 1)))
    _(Flatten())
    _(FC(shape=(1024,), activation=Relu))
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model
