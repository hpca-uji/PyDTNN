"""DenseNet121 model architecture implementation for PyDTNN."""

from collections.abc import Sequence

from pydtnn.abstract.layerable import Layerable
from pydtnn.activations.relu import Relu
from pydtnn.activations.log_softmax import LogSoftmax
from pydtnn.layers.average_pool_2d import AveragePool2D
from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.layers.concatenation_block import ConcatenationBlock
from pydtnn.layers.conv_2d import Conv2D
from pydtnn.layers.fc import FC
from pydtnn.layers.flatten import Flatten
from pydtnn.layers.input import Input
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.initializers import he_uniform

__all__ = ("densenet121",)


def densenet121(input_shape: ArrayShape, output_shape: ArrayShape) -> Sequence[Layerable]:
    """
    Constructs a DenseNet121 model architecture.

    Args:
        input_shape: The shape of the input data.
        output_shape: The shape of the output layer.

    Returns:
        A sequence of layers representing the DenseNet121 model.
    """
    model = list[Layerable]()
    _ = model.append

    _(Input(shape=input_shape))

    blocks, growth_rate = [6, 12, 24, 16], 32  # DenseNet121

    reduction = 0.5
    num_planes = 2 * growth_rate

    _(
        Conv2D(
            nfilters=num_planes,
            filter_shape=(3, 3),
            padding=1,
            use_bias=False,
            weights_initializer=he_uniform,
        )
    )

    for i, nblocks in enumerate(blocks):
        for j in range(nblocks):
            _(
                ConcatenationBlock(
                    [
                        BatchNormalization(),
                        Relu(),
                        Conv2D(
                            nfilters=4 * growth_rate,
                            filter_shape=(1, 1),
                            use_bias=False,
                            weights_initializer=he_uniform,
                        ),
                        BatchNormalization(),
                        Relu(),
                        Conv2D(
                            nfilters=growth_rate,
                            filter_shape=(3, 3),
                            padding=1,
                            use_bias=False,
                            weights_initializer=he_uniform,
                        ),
                    ],
                    [],
                )
            )

        num_planes += nblocks * growth_rate

        if i < len(blocks) - 1:
            num_planes = int(num_planes * reduction)
            _(BatchNormalization())
            _(Relu())
            _(
                Conv2D(
                    nfilters=num_planes,
                    filter_shape=(1, 1),
                    use_bias=False,
                    weights_initializer=he_uniform,
                )
            )
            _(AveragePool2D(pool_shape=(2, 2), stride=2))

    _(BatchNormalization())
    _(Relu())
    _(AveragePool2D(pool_shape=(4, 4)))
    _(Flatten())
    _(FC(shape=output_shape, activation=LogSoftmax))

    return model
