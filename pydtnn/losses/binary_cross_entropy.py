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

import numpy as np

from pydtnn.losses.loss import Loss


class BinaryCrossEntropy(Loss):

    def __call__(self, y_pred, y_targ, global_batch_size):
        assert len(y_targ.shape) == 2
        b = y_targ.shape[0]
        loss = -np.sum(np.log(np.maximum((1 - y_targ) - y_pred, self.eps))) / b
        y_pred = np.clip(y_pred, a_min=self.eps, a_max=(1 - self.eps))
        dx = (-(y_targ / y_pred) + ((1 - y_targ) / (1 - y_pred))) / global_batch_size
        return loss, dx
