from collections.abc import Sequence
from pydtnn.activations.relu import Relu
from pydtnn.activations.softmax import Softmax
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.layer_base import LayerBase

# NOTE: PyDTNN follows PyTorch's definitions
# NOTE: TensorFlow uses BatchNormalization with 1.001e-5 epsilon and 0.99 momentum
# NOTE: TensorFlow uses AveragePool2D with (2, 2) pool shape
# NOTE: TensorFlow uses FC with 1024 shape
# NOTE: TensorFlow uses LeakyReLU


def mobileNet(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerBase]:
    first_filters = 32

    model = list[LayerBase]()
    _ = model.append
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=Conv2D.Grouping.STANDARD, padding=1, stride=2, use_bias=False))
    _(BatchNormalization())
    _(Relu())

    layout = [[64, 1], [128, 2], [256, 2], [512, 6], [1024, 2]]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=Conv2D.Grouping.DEPTHWISE, padding=1, stride=stride, use_bias=False))
            _(BatchNormalization())
            _(Relu())
            _(Conv2D(nfilters=n_filt, filter_shape=(1, 1), grouping=Conv2D.Grouping.POINTWISE, use_bias=False))
            _(BatchNormalization())
            _(Relu())
            first_filters = n_filt

    _(AveragePool2D(pool_shape=(1, 1)))
    _(Flatten())
    _(FC(shape=(1024,), activation=Relu))
    _(FC(shape=output_shape, activation=Softmax))

    return model


mobilenet = mobileNet
