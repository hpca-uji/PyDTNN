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


def create_resnet101(input_shape: Sequence[int], output_shape: Sequence[int]) -> Iterable[LayerAndActivationBase]:
    yield Input(shape=input_shape)
    yield Conv2D(nfilters=64, filter_shape=(3, 3), stride=1, padding=1, weights_initializer="he_uniform")
    yield BatchNormalization()

    expansion = 4
    layout = [[64, 3, 1], [128, 4, 2], [256, 23, 2], [512, 3, 2]]  # Resnet-101
    for n_filt, res_blocks, stride in layout:
        for r in range(res_blocks):
            if r > 0:
                stride = 1
            yield AdditionBlock(
                [
                    Conv2D(nfilters=n_filt, filter_shape=(1, 1), stride=1, weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n_filt, filter_shape=(3, 3), stride=stride, padding=1,
                           weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n_filt * expansion, filter_shape=(1, 1), stride=1,
                           weights_initializer="he_uniform"),
                    BatchNormalization()
                ],
                [
                    Conv2D(nfilters=n_filt * expansion, filter_shape=(1, 1), stride=stride,
                           weights_initializer="he_uniform"),
                    BatchNormalization()
                ] if r == 0 or stride != 1 else [])
            yield Relu()

    yield AveragePool2D(pool_shape=(0, 0))  # Global average pooling 2D
    yield Flatten()
    yield FC(shape=(512 * expansion,))
    yield BatchNormalization()
    yield Relu()
    yield FC(shape=output_shape, activation="softmax")
