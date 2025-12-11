from collections.abc import Sequence
from pydtnn.activations.relu import Relu
from pydtnn.activations.relu6 import Relu6
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.dropout import Dropout
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase

# NOTE: TensorFlow uses AveragePool2D with (2, 2) pool shape


def mobileNetTF(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    """
    Mobilenet v1's TF.Keras-like version.
    """

    first_filters = 32

    epsilon = 1e-3
    momentum = 0.99

    model = list[LayerBase]()
    _ = model.append
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=Conv2D.Grouping.STANDARD, padding=0, stride=2, use_bias=False))
    _(BatchNormalization(epsilon=epsilon, momentum=momentum))
    _(Relu6())

    layout = [[64, 1], [128, 2], [256, 2], [512, 6], [1024, 2]]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=Conv2D.Grouping.DEPTHWISE, padding=1, stride=stride, use_bias=False))
            _(BatchNormalization(epsilon=epsilon, momentum=momentum))
            _(Relu6())
            _(Conv2D(nfilters=n_filt, filter_shape=(1, 1), grouping=Conv2D.Grouping.POINTWISE, use_bias=False))
            _(BatchNormalization(epsilon=epsilon, momentum=momentum))
            _(Relu6())
            first_filters = n_filt

    _(AveragePool2D(pool_shape=(1, 1)))
    _(Flatten())
    _(FC(shape=(512,), activation=Relu))
    _(Dropout(0.3))
    _(FC(shape=output_shape, activation=Softmax))

    return model


mobilenet_tf = mobileNetTF
