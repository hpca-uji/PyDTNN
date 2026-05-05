from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.relu6 import Relu6
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.conv_2d_depthwise import Conv2DDepthwise
from pydtnn.layers.conv_2d_pointwise import Conv2DPointwise
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.utils.constants import ArrayShape

# NOTE: TensorFlow uses AveragePool2D with (2, 2) pool shape


__all__ = (
    "mobileNet",
)


def mobileNet(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    first_filters = 32

    model = list[Layerable]()
    _ = model.append
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), padding=1, stride=2, use_bias=False))
    _(BatchNormalization())
    _(Relu6())

    layout = [[64, 1], [128, 2], [256, 2], [512, 6], [1024, 2]]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            _(Conv2DDepthwise(nfilters=first_filters, filter_shape=(3, 3), padding=1, stride=stride, use_bias=False))
            _(BatchNormalization())
            _(Relu6())
            _(Conv2DPointwise(nfilters=n_filt, filter_shape=(1, 1), use_bias=False))
            _(BatchNormalization())
            _(Relu6())
            first_filters = n_filt

    _(AveragePool2D((1, 1)))
    _(Flatten())
    _(FC(shape=(512,), activation=Relu6))
    _(Dropout(0.3))
    _(FC(shape=output_shape, activation=Softmax))

    return model


mobilenetv1_tensorflow = mobileNet
