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

from contextlib import suppress

from ..cython_modules import bn_relu_inference_cython
from .batch_normalization import BatchNormalization
from ..model import TRAIN_MODE


class BatchNormalizationRelu(BatchNormalization):

    def __init__(self, beta=0.0, gamma=1.0,
                 momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer="zeros",
                 moving_variance_initializer="ones",
                 sync_stats=False, from_parent=None):
        if from_parent is None:
            super(BatchNormalizationRelu, self).__init__(beta, gamma, momentum,
                                                         epsilon, moving_mean_initializer, moving_variance_initializer,
                                                         sync_stats)
        else:
            with suppress(KeyError):
                from_parent.__dict__.pop("forward")
            self.__dict__.update(from_parent.__dict__)

    def forward(self, x):
        """Version of the forward function that uses the BN + Relu"""

        if self.model.mode == TRAIN_MODE:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        if self.spatial:
            x = x.transpose(0, 2, 3, 1).reshape(-1, self.ci)

        y = bn_relu_inference_cython(x, self.running_mean, self.inv_std, self.gamma, self.beta)

        if self.spatial:
            y = y.reshape(-1, self.hi, self.wi, self.ci).transpose(0, 3, 1, 2)

        return y
