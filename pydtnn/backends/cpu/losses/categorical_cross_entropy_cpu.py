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

    def __call__(self, y_pred, y_targ, global_batch_size):
        y_pred = np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps))
        b_range = np.arange(y_pred.shape[0])
        loss = -np.sum(np.log(y_pred[b_range, np.argmax(y_targ, axis=1)])) / y_pred.shape[0]
        dx = np.copy(y_targ)
        dx_amax = np.argmax(dx, axis=1)
        dx[b_range, dx_amax] /= (-y_pred[b_range, dx_amax] * global_batch_size)
        return loss, dx
