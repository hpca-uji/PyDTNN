from collections.abc import Sequence

from pydtnn.layers import *
from pydtnn.activations import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.layers.conv_2d import GroupingEnum


def create_mobilenetv1(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    first_filters = 32
    _(Input(shape=input_shape))
    _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=GroupingEnum.STANDARD, padding=1, stride=2, activation=relu, use_bias=False))

    layout = [[64, 1], [128, 2], [256, 2], [512, 6], [1024, 2]]
    for n_filt, reps in layout:
        for r in range(reps):
            stride = 2 if reps > 1 and r == 0 else 1
            _(Conv2D(nfilters=first_filters, filter_shape=(3, 3), grouping=GroupingEnum.DEPTHWISE, padding=1, stride=stride, use_bias=False))
            _(BatchNormalization())
            _(Relu())
            _(Conv2D(nfilters=n_filt, filter_shape=(1, 1), grouping=GroupingEnum.POINTWISE, use_bias=False))
            _(BatchNormalization())
            _(Relu())
            first_filters = n_filt

    _(AveragePool2D(pool_shape=(1, 1)))
    _(Flatten())
    _(FC(shape=(1024,)))
    _(FC(shape=output_shape, activation=softmax))

    return model
