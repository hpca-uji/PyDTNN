"""MobileNetV2 implementation for the PyDTNN framework."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu6 import Relu6
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.adaptive_average_pool_2d import AdaptiveAveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.utils.constants import ArrayShape

__all__ = ("mobileNet",)


def mobileNet(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a MobileNetV2 architecture sequence.

    Args:
        input_shape: The shape of the input tensor.
        output_shape: The shape of the output tensor.

    Returns:
        A sequence of Layerable objects representing the model.
    """
    first_filters = 32
    last_channel = 1280

    model = []
    _ = model.append
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), padding=1, stride=2, use_bias=False))
    _(BatchNormalization())
    _(Relu6())

    layout = [
        # expand_ration, n_filt, reps, stride
        [1, 8, 1, 1],
        [4, 16, 1, 2],
        [4, 24, 1, 2],
        [4, 32, 1, 1],
    ]
    prev_n_filt = first_filters

    for expand_ration, n_filt, reps, _stride in layout:
        for r in range(reps):
            hidden_layers = expand_ration * prev_n_filt
            stride = _stride if r == 0 else 1
            if expand_ration != 1:
                _(
                    Conv2D(
                        nfilters=hidden_layers,
                        filter_shape=(1, 1),
                        padding=1,
                        stride=stride,
                        use_bias=False,
                    )
                )
                _(BatchNormalization())
                _(Relu6())
            # else: nothing special.

            _(
                Conv2DDepthwise(
                    nfilters=hidden_layers,
                    filter_shape=(3, 3),
                    padding=1,
                    stride=stride,
                    use_bias=False,
                )
            )
            _(BatchNormalization())
            _(Relu6())

            _(Conv2DPointwise(nfilters=n_filt, filter_shape=(1, 1), use_bias=False))
            _(BatchNormalization())
            prev_n_filt = n_filt

    _(Conv2D(nfilters=last_channel, filter_shape=(1, 1), use_bias=False))
    _(BatchNormalization())
    _(Relu6())

    _(AdaptiveAveragePool2D((1, 1)))
    _(Flatten())
    # _( FC(shape=(1024,)) )
    _(Dropout(0.2))
    _(FC(shape=output_shape, activation=Softmax))

    return model


mobilenetv2_tiny = mobileNet
