#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from collections.abc import Sequence, Iterable

from ..activations import *
from ..layers import *
from pydtnn.layers.layer_and_activation_base import LayerAndActivationBase


def create_densenet_cifar10(input_shape: Sequence[int], output_shape: Sequence[int]) -> Iterable[LayerAndActivationBase]:
    yield Input(shape= input_shape)

    blocks, growth_rate = [6, 12, 24, 16], 12

    reduction = 0.5
    num_planes = 2 * growth_rate

    yield Conv2D(nfilters=num_planes, filter_shape=(3, 3), padding=1, use_bias=False, weights_initializer="he_uniform")

    for i, nblocks in enumerate(blocks):
        for j in range(nblocks):
            yield ConcatenationBlock(
                [
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=4 * growth_rate, filter_shape=(1, 1), use_bias=False,
                           weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=growth_rate, filter_shape=(3, 3), padding=1, use_bias=False,
                           weights_initializer="he_uniform")
                ], [])

        num_planes += nblocks * growth_rate

        if i < len(blocks) - 1:
            num_planes = int(num_planes * reduction)
            yield BatchNormalization()
            yield Relu()
            yield Conv2D(nfilters=num_planes, filter_shape=(1, 1), use_bias=False, weights_initializer="he_uniform")
            yield AveragePool2D(pool_shape=(2, 2), stride=2)

    yield BatchNormalization()
    yield Relu()
    yield AveragePool2D(pool_shape=(4, 4))
    yield Flatten()
    yield FC(shape= output_shape, activation="softmax")
