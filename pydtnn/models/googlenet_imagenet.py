#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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


from ..activations import *
from ..layers import *


def create_googlenet_imagenet(model):
    _ = model.add
    _(Input(shape=(3, 224, 224)))
    _(Conv2D(nfilters=192, filter_shape=(3, 3), padding=1, weights_initializer="he_uniform"))
    _(BatchNormalization())
    _(Relu())

    inception_blocks = [[64, 96, 128, 16, 32, 32],
                        [128, 128, 192, 32, 96, 64],
                        [],
                        [192, 96, 208, 16, 48, 64],
                        [160, 112, 224, 24, 64, 64],
                        [128, 128, 256, 24, 64, 64],
                        [112, 144, 288, 32, 64, 64],
                        [256, 160, 320, 32, 128, 128],
                        [],
                        [256, 160, 320, 32, 128, 128],
                        [384, 192, 384, 48, 128, 128]]

    for layout in inception_blocks:
        if not layout:
            _(MaxPool2D(pool_shape=(3, 3), stride=2, padding=1))
        else:
            n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes = layout
            _(ConcatenationBlock(
                [
                    # 1x1 conv branch
                    Conv2D(nfilters=n1x1, filter_shape=(1, 1), weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu()
                ],
                [  # 1x1 conv -> 3x3 conv branch
                    Conv2D(nfilters=n3x3red, filter_shape=(1, 1), weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n3x3, filter_shape=(3, 3), padding=1, weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu()
                ],
                [  # 1x1 conv -> 5x5 conv branch
                    Conv2D(nfilters=n5x5red, filter_shape=(1, 1), weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n5x5, filter_shape=(3, 3), padding=1, weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu(),
                    Conv2D(nfilters=n5x5, filter_shape=(3, 3), padding=1, weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu()
                ],
                [  # 3x3 pool -> 1x1 conv branch
                    MaxPool2D(pool_shape=(3, 3), stride=1, padding=1),
                    Conv2D(nfilters=pool_planes, filter_shape=(1, 1), weights_initializer="he_uniform"),
                    BatchNormalization(),
                    Relu()
                ]))

    _(AveragePool2D(pool_shape=(8, 8), stride=1))  # Global average pooling 2D
    _(Flatten())
    _(FC(shape=(1024,)))
    _(BatchNormalization())
    _(Relu())
    _(FC(shape=(1000,), activation="softmax"))
