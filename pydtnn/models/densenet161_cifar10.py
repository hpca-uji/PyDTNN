#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

import math

from ..activations import *
from ..layers import *


def create_densenet161_cifar10(model):
    _ = model.add

    _(Input(shape=(32, 32, 3)))

    blocks, growth_rate = [6, 12, 36, 24], 48  # DenseNet161

    reduction = 0.5
    num_planes = 2 * growth_rate

    _(Conv2D(nfilters=num_planes, filter_shape=(3, 3), padding=1, use_bias=False, weights_initializer="he_uniform"))

    for i, nblocks in enumerate(blocks):
        for j in range(nblocks):
            _(ConcatenationBlock(
                [
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=4 * growth_rate, filter_shape=(1, 1), use_bias=False,
                           weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=growth_rate, filter_shape=(3, 3), padding=1, use_bias=False,
                           weights_initializer="he_uniform")
                ], []))

        num_planes += nblocks * growth_rate

        if i < len(blocks) - 1:
            num_planes = int(num_planes * reduction)
            _(BatchNormalization())
            _(Relu())
            _(Conv2D(nfilters=num_planes, filter_shape=(1, 1), use_bias=False, weights_initializer="he_uniform"))
            _(AveragePool2D(pool_shape=(2, 2), stride=2))

    _(BatchNormalization())
    _(Relu())
    _(AveragePool2D(pool_shape=(4, 4)))
    _(Flatten())
    _(FC(shape=(10,), activation="softmax"))
