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

from pydtnn.backends.cpu.metrics import MetricCPU
from pydtnn.metrics import CategoricalHinge


class CategoricalHingeCPU(MetricCPU, CategoricalHinge):

    def __call__(self, y_pred:np.ndarray, y_targ:np.ndarray) -> np.ndarray:
        #pos = np.sum(y_targ * y_pred, axis=-1)
        #neg = np.max((1.0 - y_targ) * y_pred, axis=-1)
        #return np.mean(np.maximum(0.0, neg - pos + 1), axis=-1)

        pos = y_targ * y_pred
        pos = np.sum(pos, axis=-1)        

        neg = -1 * y_targ
        neg += 1.0
        neg *= y_pred
        neg = np.max(neg, axis=-1)
                
        neg -= pos
        neg += 1
        maximum = np.maximum(0.0, neg)

        return np.mean(maximum, axis=-1)
