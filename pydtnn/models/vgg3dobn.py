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

from ..activations import *
from ..layers import *


def create_vgg3dobn(model):
    _ = model.add
    _(Input(shape=(32, 32, 3)))
    for n_filt, do_rate in zip([32, 64, 128], [0.2, 0.3, 0.4]):
        for i in range(2):
            _(Conv2D(nfilters=n_filt, filter_shape=(3, 3), padding=1, weights_initializer="he_uniform"))
            _(Relu())
            _(BatchNormalization())
        _(MaxPool2D(pool_shape=(2, 2), stride=2))
        _(Dropout(rate=do_rate))
    _(Flatten())
    _(FC(shape=(512,), weights_initializer="he_uniform"))
    _(Relu())
    _(BatchNormalization())
    _(Dropout(rate=0.5))
    _(FC(shape=(10,), weights_initializer="he_uniform"))
    _(Softmax())
