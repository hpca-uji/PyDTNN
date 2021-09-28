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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.layers import BatchNormalizationRelu
from pydtnn.cython_modules import bn_relu_inference_cython
from pydtnn.model import TRAIN_MODE
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312


class BatchNormalizationReluCPU(LayerCPU, BatchNormalizationRelu):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Version of the forward function that uses the BN + Relu"""

        if self.model.mode == TRAIN_MODE:
            raise RuntimeError("Fused layers cannot be used in training mode!")

        if self.spatial:
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
                x = best_transpose_0231(x)
            x = x.reshape(-1, self.ci)

        y = bn_relu_inference_cython(x, self.running_mean, self.inv_std, self.gamma, self.beta)

        if self.spatial:
            y = y.reshape(-1, self.hi, self.wi, self.ci)
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW:
                y = best_transpose_0312(y)

        return y

    def backward(self, x):
        raise RuntimeError(f"Backward method of {self.__class__.__name__} should not be called")
