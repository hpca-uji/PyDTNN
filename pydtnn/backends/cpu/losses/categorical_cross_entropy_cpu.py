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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from pydtnn.backends.cpu.losses.loss_cpu import LossCPU
from pydtnn.losses import CategoricalCrossEntropy


class CategoricalCrossEntropyCPU(LossCPU, CategoricalCrossEntropy):

    def __call__(self, y_pred:np.ndarray, y_targ:np.ndarray, batch_size:int) -> tuple[float, np.ndarray]:
        y_pred:np.ndarray = np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps))
        b_range:np.ndarray = np.arange(y_pred.shape[0])
        loss:float = -np.sum(np.log(y_pred[b_range, np.argmax(y_targ, axis=1)])) / y_pred.shape[0]
        dx:np.ndarray = np.copy(y_targ)
        dx_amax:np.ndarray = np.argmax(dx, axis=1)
        dx[b_range, dx_amax] /= (-y_pred[b_range, dx_amax] * batch_size)
        return loss, dx
