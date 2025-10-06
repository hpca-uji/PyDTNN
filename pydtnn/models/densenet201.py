from collections.abc import Sequence

from pydtnn.activations import *
from pydtnn.layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase
from pydtnn.initializers import he_uniform


def create_densenet201(input_shape: Sequence[int], output_shape: Sequence[int]) -> Sequence[LayerAndActivationBase]:
    model = list[LayerAndActivationBase]()
    _ = model.append

    _(Input(shape=input_shape))

    blocks, growth_rate = [6, 12, 48, 32], 32  # DenseNet201

    reduction = 0.5
    num_planes = 2 * growth_rate

    _(Conv2D(nfilters=num_planes, filter_shape=(3, 3), padding=1, use_bias=False, weights_initializer=he_uniform))

    for i, nblocks in enumerate(blocks):
        for j in range(nblocks):
            _(ConcatenationBlock(
                [
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=4 * growth_rate, filter_shape=(1, 1), use_bias=False,
                           weights_initializer=he_uniform),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=growth_rate, filter_shape=(3, 3), padding=1, use_bias=False,
                           weights_initializer=he_uniform)
                ], []))

        num_planes += nblocks * growth_rate

        if i < len(blocks) - 1:
            num_planes = int(num_planes * reduction)
            _(BatchNormalization())
            _(Relu())
            _(Conv2D(nfilters=num_planes, filter_shape=(1, 1), use_bias=False, weights_initializer=he_uniform))
            _(AveragePool2D(pool_shape=(2, 2), stride=2))

    _(BatchNormalization())
    _(Relu())
    _(AveragePool2D(pool_shape=(4, 4)))
    _(Flatten())
    _(FC(shape=output_shape, activation=softmax))

    return model
